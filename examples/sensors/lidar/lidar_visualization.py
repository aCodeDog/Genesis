#!/usr/bin/env python3
"""
LiDAR/Depth Camera Visualization and Keyboard Teleoperation

- LiDAR: shows point clouds as debug spheres
- Depth camera: shows live depth image (H x W)
"""

import argparse
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pynput import keyboard
import matplotlib.pyplot as plt

import genesis as gs
from genesis.sensors import LidarSensor
from genesis.sensors.raycaster.lidar_pattern import (
    SphericalPatternCfg,
    LivoxPatternCfg,
    SpinningLidarPatternCfg,
)
from genesis.sensors.raycaster.camera_pattern import DepthCameraPatternCfg


# ------------------------- Keyboard Helper -------------------------
class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)


# ------------------------- Scene Setup -------------------------
def build_scene(show_viewer: bool = True) -> gs.Scene:
    gs.init(backend=gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            gravity=(0.0, 0.0, -9.81),
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # Ground
    scene.add_entity(gs.morphs.Plane())

    # A ring of obstacles to visualize LiDAR hits
    inner_radius = 3.0
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Box(size=(0.3, 0.3, 1.5), pos=(x, y, 0.75), fixed=True))

    outer_radius = 5.0
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, 2.0), pos=(x, y, 1.0), fixed=True))

    return scene


def create_robot_with_lidar(
    scene: gs.Scene,
    pattern: str = "spherical",
    sensor_type: str = "mid360",
    max_range: float = 20.0,
    f_rot: Optional[float] = None,
    sample_rate: Optional[float] = None,
    n_channels: Optional[int] = None,
    # Depth camera specific
    dc_width: Optional[int] = None,
    dc_height: Optional[int] = None,
    dc_fx: Optional[float] = None,
    dc_fy: Optional[float] = None,
    dc_cx: Optional[float] = None,
    dc_cy: Optional[float] = None,
    dc_fov_h: Optional[float] = None,
    dc_fov_v: Optional[float] = None,
) -> Tuple[gs.engine.entities.RigidEntity, LidarSensor]:
    """Create fixed-base Go2 with a LiDAR or Depth Camera sensor attached."""
    base_init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    base_init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz

    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=base_init_pos, quat=base_init_quat, fixed=True)
    )

    # Choose pattern
    if pattern == "livox":
        pattern_cfg = LivoxPatternCfg(sensor_type=sensor_type)
        n_scan_lines = 1  # Livox returns flat rays; not used directly
        n_points_per_line = pattern_cfg.samples // pattern_cfg.downsample
        fov_vertical = 60.0
        fov_horizontal = 360.0
    elif pattern == "spinning":
        st = sensor_type.lower()
        # Sensible defaults per model if not provided
        if st == "vlp32":
            sample_rate = sample_rate if sample_rate is not None else 1.2e6
            n_channels = n_channels if n_channels is not None else 32
            f_rot = f_rot if f_rot is not None else 10.0
        elif st == "os128":
            sample_rate = sample_rate if sample_rate is not None else 5.2e6
            n_channels = n_channels if n_channels is not None else 128
            f_rot = f_rot if f_rot is not None else 20.0
        else:  # hdl64 default
            sample_rate = sample_rate if sample_rate is not None else 2.2e6
            n_channels = n_channels if n_channels is not None else 64
            f_rot = f_rot if f_rot is not None else 10.0
        pattern_cfg = SpinningLidarPatternCfg(
            sensor_type=st,
            f_rot=f_rot,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        n_scan_lines = 1
        # Approx number of rays in one rotation ≈ sample_rate/f_rot
        n_points_per_line = int(max(1, sample_rate / f_rot))
        fov_vertical = 45.0
        fov_horizontal = 360.0
    elif pattern == "depth":
        # Depth camera defaults
        width = int(dc_width or 640)
        height = int(dc_height or 480)
        # Optionally pick some presets by sensor_type (currently pinhole only)
        st = sensor_type.lower()
        if st not in ["pinhole", "realsense", "k4a"]:
            st = "pinhole"
        pattern_cfg = DepthCameraPatternCfg(
            width=width,
            height=height,
            fx=dc_fx,
            fy=dc_fy,
            cx=dc_cx,
            cy=dc_cy,
            fov_horizontal=dc_fov_h if dc_fov_h is not None else 90.0,
            fov_vertical=dc_fov_v,
        )
        n_scan_lines = height
        n_points_per_line = width
        # FOVs here are informational for config; rays come from intrinsics/FOV above
        fov_vertical = float(dc_fov_v) if dc_fov_v is not None else 60.0
        fov_horizontal = float(dc_fov_h) if dc_fov_h is not None else 90.0
    else:
        pattern_cfg = SphericalPatternCfg(n_scan_lines=16, n_points_per_line=64, fov_vertical=30.0, fov_horizontal=360.0)
        n_scan_lines = pattern_cfg.n_scan_lines
        n_points_per_line = pattern_cfg.n_points_per_line
        fov_vertical = pattern_cfg.fov_vertical
        fov_horizontal = pattern_cfg.fov_horizontal

    lidar = LidarSensor(
        entity=robot,
        link_idx=None,
        use_local_frame=False,
        n_scan_lines=n_scan_lines,
        n_points_per_line=n_points_per_line,
        fov_vertical=fov_vertical,
        fov_horizontal=fov_horizontal,
        max_range=max_range,
        min_range=0.1,
        pattern_cfg=pattern_cfg,
        ray_alignment="base",
        offset_pos=(0.3, 0.0, -0.06),  # mount a bit above the base
        #offset_quat_wxyz=(0.0, -0.991, -0.001, 0.131)
        
    )

    return robot, lidar


# ------------------------- Teleop + Visualization -------------------------
COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # red-ish
    (0.2, 1.0, 0.2, 1.0),  # green-ish
    (0.2, 0.6, 1.0, 1.0),  # blue-ish
    (1.0, 1.0, 0.2, 1.0),  # yellow-ish
]


def euler_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll (x), pitch (y), yaw (z) to quaternion (w, x, y, z)."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def run(scene: gs.Scene, robot, lidar: LidarSensor, n_envs: int, kb: KeyboardDevice, show_depth: bool = False, is_depth: bool = False):
    # Build scene with environments
    scene.build(n_envs=n_envs)

    print("\nKeyboard Controls:")
    print("↑/↓/←/→: Move XY, n/m: Up/Down, u/o: Roll CCW/CW, i/k: Pitch Up/Down, j/l: Yaw CCW/CW, r: Reset, esc: Quit")

    # Initial pose
    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_roll = 0.0
    init_pitch = 0.0
    init_yaw = 0.0

    target_pos = init_pos.copy()
    target_roll = init_roll
    target_pitch = init_pitch
    target_yaw = init_yaw

    # For clearing previous visualization
    point_nodes: List[Optional[object]] = [None for _ in range(n_envs)]

    # Depth image viewer
    depth_im = None
    fig = ax = None
    if show_depth:
        plt.ion()
        fig, ax = plt.subplots(num="Depth Image")
        ax.set_title("Depth (m)")

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        # Set the same pose for each environment instance
        pos_t = torch.tensor(pos_np, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        quat_t = torch.tensor(quat_np, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        for env_idx in range(n_envs):
            robot.set_pos(pos_t, envs_idx=[env_idx], zero_velocity=False)
            robot.set_quat(quat_t, envs_idx=[env_idx], zero_velocity=False)

    # Reset once at start
    apply_pose_to_all_envs(target_pos, euler_to_quat_wxyz(target_roll, target_pitch, target_yaw))

    # Main loop
    sphere_radius = 0.02
    lidar_interval = 2  # steps
    step = 0
    target_pos[2] += 0.2
    try:
        while True:
            # Handle keyboard
            pressed = kb.pressed_keys.copy()
            if keyboard.Key.esc in pressed:
                break
            if keyboard.KeyCode.from_char("r") in pressed:
                target_pos[:] = init_pos
                target_roll = init_roll
                target_pitch = init_pitch
                target_yaw = init_yaw

            # Motion increments
            dpos = 0.03
            dangle = 0.04
            if keyboard.Key.up in pressed:
                target_pos[0] += dpos
            if keyboard.Key.down in pressed:
                target_pos[0] -= dpos
            if keyboard.Key.right in pressed:
                target_pos[1] -= dpos
            if keyboard.Key.left in pressed:
                target_pos[1] += dpos
            if keyboard.KeyCode.from_char("n") in pressed:
                target_pos[2] += dpos
            if keyboard.KeyCode.from_char("m") in pressed:
                target_pos[2] -= dpos

            # Orientation increments
            if keyboard.KeyCode.from_char("u") in pressed:
                target_roll += dangle  # roll CCW around +X
            if keyboard.KeyCode.from_char("o") in pressed:
                target_roll -= dangle  # roll CW around +X
            if keyboard.KeyCode.from_char("i") in pressed:
                target_pitch += dangle  # pitch up around +Y
            if keyboard.KeyCode.from_char("k") in pressed:
                target_pitch -= dangle  # pitch down around +Y
            if keyboard.KeyCode.from_char("j") in pressed:
                target_yaw += dangle    # yaw CCW around +Z
            if keyboard.KeyCode.from_char("l") in pressed:
                target_yaw -= dangle    # yaw CW around +Z

            # Apply pose
            quat = euler_to_quat_wxyz(target_roll, target_pitch, target_yaw)
            apply_pose_to_all_envs(target_pos, quat)

            # Step physics
            scene.step()

            # Update visualization periodically
            if step % lidar_interval == 0:
                hit_points, hit_distances = lidar.read()
                # Convert to numpy on CPU
                if torch.is_tensor(hit_points):
                    hp = hit_points.detach().cpu().numpy()
                    hd = hit_distances.detach().cpu().numpy()
                else:
                    hp, hd = hit_points, hit_distances

                # hp: [n_env, S, P, 3], hd: [n_env, S, P]
                # Draw point cloud only for LiDAR patterns (not depth)
                if not is_depth:
                    for env_idx in range(n_envs):
                        valid = hd[env_idx] < lidar.config["max_range"]
                        if np.any(valid):
                            pts = hp[env_idx][valid]
                            # Clear old nodes
                            if point_nodes[env_idx] is not None:
                                scene.clear_debug_object(point_nodes[env_idx])
                            color = COLORS[env_idx % len(COLORS)]
                            # Draw spheres for visibility
                            point_nodes[env_idx] = scene.draw_debug_spheres(pts, radius=sphere_radius, color=color)

                # Depth image for env 0 (only for depth camera)
                if show_depth:
                    depth = hd[0]  # (S/H, P/W)
                    depth_disp = depth.copy()
                    depth_disp[~np.isfinite(depth_disp)] = lidar.config["max_range"]
                    depth_disp = np.clip(depth_disp, 0.0, lidar.config["max_range"])
                    if depth_im is None:
                        depth_im = ax.imshow(
                            depth_disp,
                            vmin=0.0,
                            vmax=lidar.config["max_range"],
                            cmap="plasma",
                            origin="upper",
                            aspect="auto",
                        )
                        fig.colorbar(depth_im, ax=ax)
                    else:
                        depth_im.set_data(depth_disp)
                    ax.set_xlabel("width (W)")
                    ax.set_ylabel("height/scan (H/S)")
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        for node in point_nodes:
            if node is not None:
                scene.clear_debug_object(node)
        if show_depth and plt.get_fignums():
            plt.close(fig)


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Visualization with Keyboard Teleop")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments to replicate")
    parser.add_argument("--pattern", type=str, default="livox", choices=["spherical", "livox", "spinning", "depth"], help="Sensor pattern type")
    parser.add_argument(
        "--sensor-type",
        type=str,
        default="horizon",
        choices=[
            # LiDAR
            "avia", "HAP", "horizon", "mid40", "mid70", "mid360", "tele",
            # Spinning LiDAR
            "hdl64", "vlp32", "os128",
            # Depth camera
            "pinhole", "realsense", "k4a",
        ],
        help="Sensor model (depends on --pattern)",
    )
    parser.add_argument("--max-range", type=float, default=20.0, help="Max range (m)")
    # Only meaningful for depth camera; will be ignored for LiDAR patterns
    parser.add_argument("--show-depth", action="store_true", help="Show depth image window (only for --pattern depth)")

    # Spinning-specific optional overrides
    parser.add_argument("--f-rot", type=float, default=None, help="Spinning lidar rotation frequency (Hz)")
    parser.add_argument("--sample-rate", type=float, default=None, help="Spinning lidar sample rate (samples/sec)")
    parser.add_argument("--n-channels", type=int, default=None, help="Spinning lidar channel count")

    # Depth camera options
    parser.add_argument("--dc-width", type=int, default=None, help="Depth camera image width")
    parser.add_argument("--dc-height", type=int, default=None, help="Depth camera image height")
    parser.add_argument("--dc-fx", type=float, default=None, help="Depth camera fx (pixels)")
    parser.add_argument("--dc-fy", type=float, default=None, help="Depth camera fy (pixels)")
    parser.add_argument("--dc-cx", type=float, default=None, help="Depth camera cx (pixels)")
    parser.add_argument("--dc-cy", type=float, default=None, help="Depth camera cy (pixels)")
    parser.add_argument("--dc-fov-h", type=float, default=None, help="Depth camera horizontal FOV (deg)")
    parser.add_argument("--dc-fov-v", type=float, default=None, help="Depth camera vertical FOV (deg)")

    args = parser.parse_args()

    # Enforce show-depth only for depth camera
    is_depth = (args.pattern == "depth")
    show_depth = bool(args.show_depth and is_depth)
    if args.show_depth and not is_depth:
        print("[info] --show-depth is only used for --pattern depth. Ignoring for LiDAR patterns.")

    kb = KeyboardDevice()
    kb.start()

    scene = build_scene(show_viewer=True)
    robot, lidar = create_robot_with_lidar(
        scene,
        pattern=args.pattern,
        sensor_type=args.sensor_type,
        max_range=args.max_range,
        f_rot=args.f_rot,
        sample_rate=args.sample_rate,
        n_channels=args.n_channels,
        dc_width=args.dc_width,
        dc_height=args.dc_height,
        dc_fx=args.dc_fx,
        dc_fy=args.dc_fy,
        dc_cx=args.dc_cx,
        dc_cy=args.dc_cy,
        dc_fov_h=args.dc_fov_h,
        dc_fov_v=args.dc_fov_v,
    )

    run(scene, robot, lidar, n_envs=args.n_envs, kb=kb, show_depth=show_depth, is_depth=is_depth)


if __name__ == "__main__":
    main()
