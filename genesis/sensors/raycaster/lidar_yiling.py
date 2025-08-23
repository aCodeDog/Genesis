from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities import RigidEntity
from genesis.engine.bvh import AABB, LBVH
from genesis.utils.geom import (
    ti_normalize,
    quat_to_R,
)
from genesis.utils.misc import tensor_to_array
import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp


from ..base_sensor import Sensor
from .lidar_pattern import (
    PatternBaseCfg, SphericalPatternCfg, LidarPatternCfg, BpearlPatternCfg, GridPatternCfg, LivoxPatternCfg,
    generate_ray_pattern, create_spherical_pattern, create_pattern_generator, generate_ray_pattern_with_starts
)

MapLidarFaces = ti.template()


@ti.func
def ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2):
    """
    Möller-Trumbore ray-triangle intersection.

    Returns: vec4(t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    # Declare all variables at the top to avoid scope issues
    t = 0.0
    u = 0.0
    v = 0.0
    f = 0.0
    s = ti.math.vec3(0.0, 0.0, 0.0)
    q = ti.math.vec3(0.0, 0.0, 0.0)

    # If determinant is near zero, ray lies in plane of triangle
    if ti.abs(a) < 1e-8:
        valid = False

    if valid:
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)

        # Check u parameter bounds
        if u < 0.0 or u > 1.0:
            valid = False

    if valid:
        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        # Check v parameter bounds
        if v < 0.0 or u + v > 1.0:
            valid = False

    if valid:
        # At this stage we can compute t to find out where the intersection point is on the line
        t = f * edge2.dot(q)

        # Ray intersection
        if t <= 1e-8:  # Invalid intersection
            valid = False

    # Set result only if valid
    if valid:
        result = ti.math.vec4(t, u, v, 1.0)

    return result


@ti.func
def ray_aabb_intersection(ray_start, ray_dir, aabb_min, aabb_max):
    """
    Fast ray-AABB intersection test.
    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection
    inv_dir = 1.0 / ray_dir

    # Handle potential division by zero with large values
    if ti.abs(ray_dir.x) < 1e-10:
        inv_dir.x = 1e10 if ray_dir.x >= 0.0 else -1e10
    if ti.abs(ray_dir.y) < 1e-10:
        inv_dir.y = 1e10 if ray_dir.y >= 0.0 else -1e10
    if ti.abs(ray_dir.z) < 1e-10:
        inv_dir.z = 1e10 if ray_dir.z >= 0.0 else -1e10

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = ti.min(t1, t2)
    tmax = ti.max(t1, t2)

    t_near = ti.max(ti.max(tmin.x, tmin.y), tmin.z)
    t_far = ti.min(ti.min(tmax.x, tmax.y), tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far and t_far >= 0.0:
        result = ti.max(t_near, 0.0)

    return result


@ti.kernel
def kernel_update_aabbs(
    map_lidar_faces: MapLidarFaces,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    aabb_state: array_class.AABBState,
):
    _B = free_verts_state.pos.shape[1]
    # n_faces = faces_info.geom_idx.shape[0]
    n_faces = map_lidar_faces.shape[0]
    # step 1: update free verts
    for i_b, i_f_ in ti.ndrange(_B, n_faces):
        i_f = map_lidar_faces[i_f_]
        aabb_state.aabbs[i_b, i_f].min.fill(np.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-np.inf)

        is_free = verts_info.is_free[faces_info.verts_idx[i_f][0]]
        if is_free:
            for i in ti.static(range(3)):
                i_v = verts_info.verts_state_idx[faces_info.verts_idx[i_f][i]]
                pos_v = free_verts_state.pos[i_v, i_b]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)

        elif i_b == 0:  #
            for i in ti.static(range(3)):
                i_v = verts_info.verts_state_idx[faces_info.verts_idx[i_f][i]]
                pos_v = fixed_verts_state.pos[i_v]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@ti.kernel
def kernel_cast_rays_bvh(
    map_lidar_faces: MapLidarFaces,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    # BVH data structures
    bvh_nodes: ti.template(),  # The BVH node tree
    bvh_morton_codes: ti.template(),  # Maps sorted leaves to original triangle indices
    # Per-ray data (precomputed, world frame)
    ray_starts_world: ti.types.ndarray(ndim=5),  # [n_env, n_cam, n_scan_lines, n_points, 3]
    ray_directions_world: ti.types.ndarray(ndim=5),  # [n_env, n_cam, n_scan_lines, n_points, 3]
    # Optional local directions for local-frame output
    ray_directions_local: ti.types.ndarray(ndim=3),  # [n_scan_lines, n_points, 3]
    far_plane: ti.f32,
    # Output arrays
    hit_points: ti.types.ndarray(ndim=5),  # [n_env, n_cam, n_scan_lines, n_points, 3]
    hit_distances: ti.types.ndarray(ndim=4),  # [n_env, n_cam, n_scan_lines, n_points]
    world_frame: ti.i32,
):
    """
    Taichi kernel for LiDAR ray casting, accelerated by a Bounding Volume Hierarchy (BVH).
    """
    n_triangles = map_lidar_faces.shape[0]
    # Parallel execution over all rays
    for env_id, cam_id, scan_line, point_index in ti.ndrange(
        hit_points.shape[0], hit_points.shape[1], hit_points.shape[2], hit_points.shape[3]
    ):
        # --- 1. Setup Ray (already in world frame) ---
        ray_start_world = ti.math.vec3(
            ray_starts_world[env_id, cam_id, scan_line, point_index, 0],
            ray_starts_world[env_id, cam_id, scan_line, point_index, 1],
            ray_starts_world[env_id, cam_id, scan_line, point_index, 2],
        )
        ray_direction_world = ti_normalize(
            ti.math.vec3(
                ray_directions_world[env_id, cam_id, scan_line, point_index, 0],
                ray_directions_world[env_id, cam_id, scan_line, point_index, 1],
                ray_directions_world[env_id, cam_id, scan_line, point_index, 2],
            )
        )

        # --- 2. BVH Traversal ---
        min_t = far_plane
        hit_face = -1

        # Stack for non-recursive traversal, size 64 is typical for BVH
        stack = ti.Vector.zero(ti.i32, 64)
        stack[0] = 0  # Start traversal at the root node (index 0)
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr]

            # Since n_batches=1, we index the BVH with [0, node_idx]
            node = bvh_nodes[0, node_idx]

            # Check if ray hits the node's bounding box
            aabb_t = ray_aabb_intersection(ray_start_world, ray_direction_world, node.bound.min, node.bound.max)

            if aabb_t >= 0.0 and aabb_t < min_t:
                if node.left == -1:  # It's a LEAF node
                    # A leaf node corresponds to one of the sorted triangles.
                    # We need to find the original triangle index.
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    original_tri_idx = bvh_morton_codes[0, sorted_leaf_idx][1]

                    i_f = map_lidar_faces[original_tri_idx]
                    is_free = verts_info.is_free[faces_info.verts_idx[i_f][0]]

                    v0 = ti.Vector.zero(gs.ti_float, 3)
                    v1 = ti.Vector.zero(gs.ti_float, 3)
                    v2 = ti.Vector.zero(gs.ti_float, 3)

                    if is_free:
                        v0 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]], env_id]
                        v1 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]], env_id]
                        v2 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]], env_id]

                    else:
                        v0 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]]]
                        v1 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]]]
                        v2 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]]]

                    # Perform the expensive ray-triangle intersection test
                    hit_result = ray_triangle_intersection(ray_start_world, ray_direction_world, v0, v1, v2)

                    if hit_result.w > 0.0 and hit_result.x < min_t and hit_result.x >= 0.0:
                        min_t = hit_result.x
                        hit_face = i_f
                        # hit_u, hit_v could be stored here if needed

                else:  # It's an INTERNAL node
                    # Push children onto the stack for further traversal
                    # Make sure stack doesn't overflow
                    if stack_ptr < 62:
                        stack[stack_ptr] = node.left
                        stack[stack_ptr + 1] = node.right
                        stack_ptr += 2

        # --- 3. Process Hit Result ---
        if hit_face >= 0:
            dist = min_t
            hit_distances[env_id, cam_id, scan_line, point_index] = dist

            if world_frame:
                hit_point = ray_start_world + dist * ray_direction_world
                hit_points[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                hit_points[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                hit_points[env_id, cam_id, scan_line, point_index, 2] = hit_point.z
            else:
                # Local frame output along provided local ray direction
                hit_point = dist * ti_normalize(
                    ti.math.vec3(
                        ray_directions_local[scan_line, point_index, 0],
                        ray_directions_local[scan_line, point_index, 1],
                        ray_directions_local[scan_line, point_index, 2],
                    )
                )
                hit_points[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                hit_points[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                hit_points[env_id, cam_id, scan_line, point_index, 2] = hit_point.z

        else:
            hit_distances[env_id, cam_id, scan_line, point_index] = 1000.0
            hit_points[env_id, cam_id, scan_line, point_index, 0] = 0.0
            hit_points[env_id, cam_id, scan_line, point_index, 1] = 0.0
            hit_points[env_id, cam_id, scan_line, point_index, 2] = 0.0


@ti.data_oriented
class LidarSensor(Sensor):
    """
    Taichi-accelerated LiDAR sensor using BVH traversal.
    - Supports multiple ray patterns with pattern_cfg.
    - Mounting offset and alignment modes: world, yaw, base.
    - Optional dynamic Livox updates.
    """

    _mesh_registered = False
    _kernels = None
    _scene_geometry_cache = None
    _scene_mesh_info = None  # Store detailed mesh information
    _scene_mesh_data = None  # Store original mesh data for static extraction

    # TODO: Future API for dynamic mesh support
    # _dynamic_entities = []  # List of entities that can move
    # _update_dynamic_mesh = False  # Flag to update dynamic entities

    def __init__(
        self,
        entity: RigidEntity,
        link_idx: Optional[int] = None,
        use_local_frame: bool = False,
        n_scan_lines: int = 32,
        n_points_per_line: int = 64,
        fov_vertical: float = 30.0,
        fov_horizontal: float = 360.0,
        max_range: float = 20.0,
        min_range: float = 0.1,
        only_cast_fixed: bool = False,
        pattern_cfg: Optional[PatternBaseCfg] = None,
        # New: mounting offset and alignment + drift like Warp
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        ray_alignment: str = "base",  # one of {"world","yaw","base"}
    ):

        self._entity = entity
        self._sim = entity._sim
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx
        self._use_local_frame = use_local_frame

        self.config = {
            "n_scan_lines": n_scan_lines,
            "n_points_per_line": n_points_per_line,
            "fov_vertical": fov_vertical,
            "fov_horizontal": fov_horizontal,
            "max_range": max_range,
            "min_range": min_range,
        }

        self.only_cast_fixed = only_cast_fixed

        # Store pattern configuration
        self.pattern_cfg = pattern_cfg

        # Initialize pattern generator for dynamic patterns
        self.pattern_generator = None
        if self.pattern_cfg and isinstance(self.pattern_cfg, LivoxPatternCfg):
            self.pattern_generator = create_pattern_generator(self.pattern_cfg)

        # Mounting offset and alignment
        self.offset_pos = torch.tensor(offset_pos, dtype=gs.tc_float, device=gs.device)
        self.offset_quat_wxyz = torch.tensor(offset_quat_wxyz, dtype=gs.tc_float, device=gs.device)
        self.ray_alignment = ray_alignment

        # Generate ray pattern (local starts + dirs), then apply offset
        self._init_local_rays_with_offset()

        # Drift buffers (world drift on origin; per-ray-cast drift in sensor frame)
        # Defer allocation until solver is built
        self.drift = None
        self.ray_cast_drift = None

        # Dynamic pattern tracking
        self.simulation_time = 0.0
        self.last_pattern_update_time = 0.0

        # build bvh
        self.solver = self._sim.rigid_solver
        self.is_built = False

    def _init_local_rays_with_offset(self):
        # Get local starts and directions in sensor frame
        if self.pattern_cfg is not None:
            starts_np, dirs_np = generate_ray_pattern_with_starts(self.pattern_cfg)
        else:
            # default spherical
            dirs_np = create_spherical_pattern(
                n_scan_lines=self.config["n_scan_lines"],
                n_points_per_line=self.config["n_points_per_line"],
                fov_vertical=self.config["fov_vertical"],
                fov_horizontal=self.config["fov_horizontal"],
            )
            starts_np = np.zeros_like(dirs_np, dtype=np.float32)

        # Convert to torch
        self.ray_starts_local = torch.tensor(starts_np, dtype=gs.tc_float, device=gs.device)  # [S,P,3]
        self.ray_dirs_local = torch.tensor(dirs_np, dtype=gs.tc_float, device=gs.device)      # [S,P,3]

        # Apply mounting offset: rotate directions, translate starts
        # Build rotation matrix from offset quaternion (wxyz)
        R_off = quat_to_R(self.offset_quat_wxyz.view(1, 4))[0]  # [3,3]
        self.ray_dirs_local = torch.einsum('ij,spj->spi', R_off, self.ray_dirs_local)
        self.ray_starts_local = self.ray_starts_local + self.offset_pos.view(1, 1, 3)

    def filter_lidar_faces(self):
        n_lidar_faces = self.solver.faces_info.geom_idx.shape[0]
        np_map_lidar_faces = np.arange(n_lidar_faces)
        if self.only_cast_fixed:
            # count the number of faces in a fixed geoms
            geom_is_fixed = np.logical_not(self.solver.geoms_info.is_free.to_numpy())
            faces_geom = self.solver.faces_info.geom_idx.to_numpy()
            n_lidar_faces = np.sum(geom_is_fixed[faces_geom])
            np_map_lidar_faces = np.where(geom_is_fixed[faces_geom])[0]
        # from IPython import embed; embed()
        return n_lidar_faces, np_map_lidar_faces

    def build(self):
        n_lidar_faces, np_map_lidar_faces = self.filter_lidar_faces()

        self.n_lidar_faces = n_lidar_faces
        self.map_lidar_faces = ti.field(ti.i32, (n_lidar_faces))
        self.map_lidar_faces.from_numpy(np_map_lidar_faces)

        self.aabbs = AABB(n_batches=self.solver.free_verts_state.pos.shape[1], n_aabbs=self.n_lidar_faces)

        rigid_solver_decomp.kernel_update_all_verts(
            geoms_state=self.solver.geoms_state,
            verts_info=self.solver.verts_info,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
        )

        kernel_update_aabbs(
            map_lidar_faces=self.map_lidar_faces,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            aabb_state=self.aabbs,
        )

        self.bvh = LBVH(self.aabbs)
        self.bvh.build()

        # Allocate drift buffers now that we know n_envs
        n_envs = self.solver.free_verts_state.pos.shape[1]
        if self.drift is None or self.drift.shape[0] != n_envs:
            self.drift = torch.zeros((n_envs, 3), dtype=gs.tc_float, device=gs.device)
            self.ray_cast_drift = torch.zeros((n_envs, 3), dtype=gs.tc_float, device=gs.device)

    def _create_ray_pattern(self) -> np.ndarray:
        """Create LiDAR ray pattern based on configuration.
        Deprecated by _init_local_rays_with_offset for new flow; kept for compatibility."""
        if self.pattern_cfg is not None:
            return generate_ray_pattern(self.pattern_cfg)
        else:
            return create_spherical_pattern(
                n_scan_lines=self.config["n_scan_lines"],
                n_points_per_line=self.config["n_points_per_line"],
                fov_vertical=self.config["fov_vertical"],
                fov_horizontal=self.config["fov_horizontal"]
            )

    def _rotate_yaw_only(self, quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vec by yaw (about z) extracted from quat. Supports broadcasting.
        quat_wxyz: [B,4], vec: [...,3] where leading dim matches B via broadcasting rules.
        Returns vec rotated only in XY plane and preserves z component.
        """
        # Extract yaw from quaternion
        w, x, y, z = quat_wxyz.unbind(-1)
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2)) (assuming wxyz)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        vx = vec[..., 0]
        vy = vec[..., 1]
        vz = vec[..., 2]
        rx = cy * vx - sy * vy
        ry = sy * vx + cy * vy
        return torch.stack([rx, ry, vz], dim=-1)

    def _quat_rotate(self, quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vec by full quaternion using rotation matrices. Supports broadcasting.
        quat_wxyz: [B,4], vec: [B,S,P,3] or [B,3]."""
        R = quat_to_R(quat_wxyz)  # [B,3,3]
        if vec.dim() == 2:
            return torch.einsum('bij,bj->bi', R, vec)
        elif vec.dim() == 4:
            return torch.einsum('bij,bspj->bspi', R, vec)
        else:
            # [S,P,3] with single R per-batch not supported here
            raise ValueError("Unsupported vec shape for _quat_rotate")

    def read(self, envs_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the LiDAR sensor data.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            hit_points: Hit points array [n_env, n_scan_lines, n_points, 3]
            hit_distances: Hit distances array [n_env, n_scan_lines, n_points]
        """
        if not self.is_built:
            self.build()
            self.is_built = True

        # Optional dynamic Livox pattern update
        if self.pattern_cfg is not None and isinstance(self.pattern_cfg, LivoxPatternCfg) and self.pattern_generator:
            updated = self.pattern_generator.update_dynamic_pattern(self.pattern_cfg, self.simulation_time)
            if updated is not None:
                # updated shape [1,N,3] or [S,P,3]; we keep starts same, replace dirs
                updated_dirs = torch.tensor(updated, dtype=gs.tc_float, device=gs.device)
                # Apply offset to new dirs
                R_off = quat_to_R(self.offset_quat_wxyz.view(1, 4))[0]
                updated_dirs = torch.einsum('ij,spj->spi', R_off, updated_dirs)
                self.ray_dirs_local = updated_dirs

        if not self.only_cast_fixed:
            rigid_solver_decomp.kernel_update_all_verts(
                geoms_state=self.solver.geoms_state,
                verts_info=self.solver.verts_info,
                free_verts_state=self.solver.free_verts_state,
                fixed_verts_state=self.solver.fixed_verts_state,
            )

            kernel_update_aabbs(
                map_lidar_faces=self.map_lidar_faces,
                free_verts_state=self.solver.free_verts_state,
                fixed_verts_state=self.solver.fixed_verts_state,
                verts_info=self.solver.verts_info,
                faces_info=self.solver.faces_info,
                aabb_state=self.aabbs,
            )

        n_envs = self.solver.free_verts_state.pos.shape[1]
        S, P = self.ray_dirs_local.shape[:2]

        # Ensure drift buffers exist and match n_envs
        if self.drift is None or self.drift.shape[0] != n_envs:
            self.drift = torch.zeros((n_envs, 3), dtype=gs.tc_float, device=gs.device)
            self.ray_cast_drift = torch.zeros((n_envs, 3), dtype=gs.tc_float, device=gs.device)

        # Prepare output arrays
        hit_points = torch.zeros(size=(n_envs, 1, S, P, 3), dtype=gs.tc_float, device=gs.device)
        hit_distances = torch.zeros(size=(n_envs, 1, S, P), dtype=gs.tc_float, device=gs.device)

        # Get current LiDAR poses
        lidar_positions = self.solver.get_links_pos(links_idx=self.link_idx).squeeze(axis=1).reshape(n_envs, 3)
        lidar_quaternions = self.solver.get_links_quat(links_idx=self.link_idx).squeeze(axis=1).reshape(n_envs, 4)  # wxyz

        # Apply world drift to origin
        lidar_positions = lidar_positions + self.drift

        # Build world rays per alignment
        # Expand for broadcasting [B,1,S,P,3]
        pos_w_exp = lidar_positions.view(n_envs, 1, 1, 1, 3)
        # Ray starts/directions local
        starts_local = self.ray_starts_local  # [S,P,3]
        dirs_local = self.ray_dirs_local      # [S,P,3]

        if self.ray_alignment == "world":
            # apply horizontal drift in ray caster frame (XY)
            pos_w = lidar_positions.clone()
            pos_w[:, 0:2] += self.ray_cast_drift[:, 0:2]
            pos_w_exp = pos_w.view(n_envs, 1, 1, 1, 3)
            ray_starts_world = starts_local.view(1, 1, S, P, 3).expand(n_envs, 1, S, P, 3) + pos_w_exp
            ray_dirs_world = dirs_local.view(1, 1, S, P, 3).expand(n_envs, 1, S, P, 3)
        elif self.ray_alignment == "yaw":
            # yaw rotate starts; directions unchanged
            yaw_rot_starts = self._rotate_yaw_only(lidar_quaternions, starts_local)  # -> [B,S,P,3]
            pos_w = lidar_positions.clone()
            drift_yaw = self._rotate_yaw_only(lidar_quaternions, self.ray_cast_drift)
            pos_w[:, 0:2] += drift_yaw[:, 0:2]
            pos_w_exp = pos_w.view(n_envs, 1, 1, 1, 3)
            ray_starts_world = yaw_rot_starts.view(n_envs, 1, S, P, 3) + pos_w_exp
            ray_dirs_world = dirs_local.view(1, 1, S, P, 3).expand(n_envs, 1, S, P, 3)
        elif self.ray_alignment == "base":
            # full rotation for starts and directions
            starts_batched = starts_local.view(1, S, P, 3).expand(n_envs, S, P, 3)
            dirs_batched = dirs_local.view(1, S, P, 3).expand(n_envs, S, P, 3)
            rot_starts = self._quat_rotate(lidar_quaternions, starts_batched)
            rot_dirs = self._quat_rotate(lidar_quaternions, dirs_batched)
            pos_w = lidar_positions.clone()
            drift_base = self._quat_rotate(lidar_quaternions, self.ray_cast_drift)
            pos_w[:, 0:2] += drift_base[:, 0:2]
            pos_w_exp = pos_w.view(n_envs, 1, 1, 1, 3)
            ray_starts_world = rot_starts.view(n_envs, 1, S, P, 3) + pos_w_exp
            ray_dirs_world = rot_dirs.view(n_envs, 1, S, P, 3)
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.ray_alignment}")

        # Cast rays
        kernel_cast_rays_bvh(
            map_lidar_faces=self.map_lidar_faces,
            fixed_verts_state=self.solver.fixed_verts_state,
            free_verts_state=self.solver.free_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            bvh_nodes=self.bvh.nodes,
            bvh_morton_codes=self.bvh.morton_codes,
            ray_starts_world=ray_starts_world.contiguous(),
            ray_directions_world=ray_dirs_world.contiguous(),
            ray_directions_local=self.ray_dirs_local.contiguous(),
            far_plane=self.config["max_range"],
            hit_points=hit_points,
            hit_distances=hit_distances,
            world_frame=0 if self._use_local_frame else 1,
        )

        # Remove the camera dimension (we only have 1 camera per sensor)
        hit_points = hit_points.squeeze(1)  # [n_env, S, P, 3]
        hit_distances = hit_distances.squeeze(1)  # [n_env, S, P]

        # Apply post-cast vertical drift (z)
        hit_points[..., 2] += self.ray_cast_drift[:, 2].view(n_envs, 1, 1)

        # Return requested subset
        if envs_idx is not None:
            return hit_points[envs_idx], hit_distances[envs_idx]
        else:
            return hit_points, hit_distances

    def get_point_cloud(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the point cloud from the LiDAR sensor.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            Point cloud array [n_env, n_points, 3] where n_points = n_scan_lines * n_points_per_line
        """
        hit_points, hit_distances = self.read(envs_idx)

        if hit_points is None:
            return None

        # Filter out invalid points (beyond max range)
        valid_mask = hit_distances < self.config["max_range"]

        # Reshape to flat point cloud
        n_envs = hit_points.shape[0]
        n_total_points = hit_points.shape[1] * hit_points.shape[2]

        point_cloud = hit_points.reshape(n_envs, n_total_points, 3)
        valid_mask = valid_mask.reshape(n_envs, n_total_points)

        # Zero out invalid points
        point_cloud[~valid_mask] = 0.0

        return point_cloud

    def get_distances(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the distance measurements from the LiDAR sensor.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            Distance array [n_env, n_scan_lines, n_points]
        """
        _, hit_distances = self.read(envs_idx)
        return hit_distances

    @property
    def n_scan_lines(self) -> int:
        """Number of vertical scan lines."""
        return self.config["n_scan_lines"]

    @property
    def n_points_per_line(self) -> int:
        """Number of horizontal points per scan line."""
        return self.config["n_points_per_line"]

    @property
    def max_range(self) -> float:
        """Maximum sensing range."""
        return self.config["max_range"]

    @property
    def min_range(self) -> float:
        """Minimum sensing range."""
        return self.config["min_range"]

    def get_mesh_info(self) -> Optional[Dict]:
        """
        Get detailed information about the extracted scene mesh.

        Returns:
            Dictionary containing mesh information:
            - entities: List of entity information with geometry details
            - total_vertices: Total number of vertices in the scene
            - total_triangles: Total number of triangles in the scene
            - geometry_count: Number of geometries processed
            - extraction_time: Time taken to extract meshes (in milliseconds)
        """
        print("TODO: get_mesh_info not implemented")
        return
        # Ensure mesh is extracted
        self._extract_scene_geometry()
        return LidarSensor._scene_mesh_info

    def get_scene_mesh(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the combined scene mesh data used for ray casting.

        Returns:
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        print("TODO: get_scene_mesh not implemented")
        return
        try:
            return self._extract_scene_geometry()
        except Exception as e:
            print(f"Error extracting scene mesh: {e}")
            return None, None

    def save_scene_mesh(self, filepath: str) -> bool:
        """
        Save the extracted scene mesh to a file.

        Args:
            filepath: Path where to save the mesh (supports .obj, .ply, .stl formats)

        Returns:
            True if successful, False otherwise
        """
        print("TODO: save_scene_mesh not implemented")
        return
        try:
            vertices, triangles = self._extract_scene_geometry()
            if vertices is not None and triangles is not None:
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                mesh.export(filepath)
                print(f"LiDAR: Scene mesh saved to {filepath}")
                return True
            else:
                print("LiDAR: No scene mesh available to save")
                return False
        except Exception as e:
            print(f"LiDAR: Error saving scene mesh to {filepath}: {e}")
            return False

    def print_mesh_summary(self):
        print("TODO: print_mesh_summary not implemented")
        return
        """Print a summary of the extracted scene mesh information."""
        mesh_info = self.get_mesh_info()
        if mesh_info is None:
            print("LiDAR: No mesh information available")
            return

        print("=== LiDAR Scene Mesh Summary ===")
        print(f"Total vertices: {mesh_info['total_vertices']}")
        print(f"Total triangles: {mesh_info['total_triangles']}")
        print(f"Geometry count: {mesh_info['geometry_count']}")
        print(f"Entity count: {len(mesh_info['entities'])}")
        print(f"Extraction time: {mesh_info['extraction_time']:.1f}ms")
        print()

        for i, entity_info in enumerate(mesh_info["entities"]):
            print(f"Entity {i+1} ({entity_info['type']}):")
            print(f"  Vertices: {entity_info['vertex_count']}")
            print(f"  Triangles: {entity_info['triangle_count']}")
            print(f"  Geometries: {len(entity_info['geometries'])}")

            for j, geom_info in enumerate(entity_info["geometries"]):
                print(f"    Geometry {j+1}: {geom_info['vertices']} vertices, {geom_info['triangles']} triangles")
        print("================================")

    # TODO: Future API for dynamic mesh support
    def add_dynamic_entity(self, entity):
        """
        Add an entity to the dynamic mesh tracking list (future feature).

        Args:
            entity: Entity that can move and needs mesh updates
        """
        raise NotImplementedError("Dynamic mesh support not yet implemented")

    def update_dynamic_meshes(self):
        """
        Update meshes for dynamic entities (future feature).
        This would re-transform only the meshes of entities that have moved.
        """
        raise NotImplementedError("Dynamic mesh support not yet implemented")

    def set_static_mode(self, static: bool = True):
        """
        Toggle between static and dynamic mesh modes (future feature).

        Args:
            static: If True, use static mesh (current behavior).
                   If False, enable dynamic mesh updates.
        """
        if not static:
            raise NotImplementedError("Dynamic mesh support not yet implemented")
        # Static mode is always enabled for now
