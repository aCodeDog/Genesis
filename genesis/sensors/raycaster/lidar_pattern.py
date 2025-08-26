"""
Taichi-based LiDAR pattern generation for Genesis.

This module provides various ray patterns for LiDAR sensors, implemented using Taichi
for efficient computation. It mirrors the functionality of the Warp-based patterns
but uses Taichi for computation.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Sequence, Dict
from dataclasses import dataclass, field
import hashlib


@dataclass
class PatternBaseCfg:
    """Base configuration for a pattern."""
    pass


@dataclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.
    
    Defines a 2D grid of rays in the coordinates of the sensor.
    """
    resolution: float = 0.1  # Grid resolution (in meters)
    size: Tuple[float, float] = (2.0, 2.0)  # Grid size (length, width) (in meters)
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)  # Ray direction
    ordering: str = "xy"  # Ordering of points: "xy" or "yx"


@dataclass
class LidarPatternCfg(PatternBaseCfg):
    """Configuration for the LiDAR pattern for ray-casting."""
    channels: int = 32  # Number of vertical channels (beams)
    vertical_fov_range: Tuple[float, float] = (-15.0, 15.0)  # Vertical FOV in degrees
    horizontal_fov_range: Tuple[float, float] = (-180.0, 180.0)  # Horizontal FOV in degrees
    horizontal_res: float = 1.0  # Horizontal resolution in degrees


@dataclass
class BpearlPatternCfg(PatternBaseCfg):
    """Configuration for the Bpearl pattern for ray-casting."""
    horizontal_fov: float = 360.0  # Horizontal field of view (in degrees)
    horizontal_res: float = 10.0  # Horizontal resolution (in degrees)
    vertical_ray_angles: List[float] = field(default_factory=lambda: [
        89.5, 86.6875, 83.875, 81.0625, 78.25, 75.4375, 72.625, 69.8125, 67.0, 64.1875, 61.375,
        58.5625, 55.75, 52.9375, 50.125, 47.3125, 44.5, 41.6875, 38.875, 36.0625, 33.25, 30.4375,
        27.625, 24.8125, 22, 19.1875, 16.375, 13.5625, 10.75, 7.9375, 5.125, 2.3125
    ])


@dataclass
class SphericalPatternCfg(PatternBaseCfg):
    """Configuration for spherical uniform pattern for ray-casting."""
    n_scan_lines: int = 32  # Number of vertical scan lines
    n_points_per_line: int = 64  # Number of horizontal points per scan line
    fov_vertical: float = 30.0  # Vertical field of view in degrees
    fov_horizontal: float = 360.0  # Horizontal field of view in degrees


@dataclass
class LivoxPatternCfg(PatternBaseCfg):
    """Configuration for Livox LiDAR pattern for ray-casting."""
    sensor_type: str = "avia"  # Type of Livox sensor
    samples: int = 24000  # Number of ray samples per scan frame
    downsample: int = 1  # Downsampling factor for ray patterns
    use_simple_grid: bool = False  # Whether to use simple grid pattern instead
    rolling_window_start: int = 0  # Starting index for rolling window sampling
    
    # Simple grid parameters (used when use_simple_grid=True)
    horizontal_line_num: int = 80
    vertical_line_num: int = 50
    horizontal_fov_deg_min: float = -180
    horizontal_fov_deg_max: float = 180
    vertical_fov_deg_min: float = -2
    vertical_fov_deg_max: float = 57
    
    # Dynamic pattern parameters
    enable_dynamic_pattern: bool = True  # Enable dynamic ray updates
    pattern_rotation_speed: float = 0.1  # Rotation speed for dynamic patterns


@dataclass
class SpinningLidarPatternCfg(PatternBaseCfg):
    """Configuration for traditional spinning lidars (HDL64, VLP32, OS128)."""
    sensor_type: str = "hdl64"  # one of {"hdl64", "vlp32", "os128"}
    f_rot: float = 10.0          # rotation frequency (Hz)
    sample_rate: float = 2.2e6   # samples per second (defaults for HDL64)
    n_channels: int = 64         # number of channels (64/32/128)
    phi_fov: Tuple[float, float] = (-24.9, 2.0)  # deg, used for HDL64 when no custom table


class PatternGenerator:
    """Base class for pattern generators using Taichi."""
    
    def __init__(self):
        self._ray_vectors = None
        self._n_rays = 0
        
    @abstractmethod
    def generate_pattern(self, cfg: PatternBaseCfg) -> np.ndarray:
        """Generate ray pattern based on configuration.
        
        Args:
            cfg: Pattern configuration
            
        Returns:
            Ray vectors array of shape [n_scan_lines, n_points_per_line, 3] or [n_rays, 3]
        """
        pass


class GridPatternGenerator(PatternGenerator):
    """Grid pattern generator using Taichi."""
    
    def generate_pattern(self, cfg: GridPatternCfg) -> np.ndarray:
        """Generate grid pattern."""
        if cfg.ordering not in ["xy", "yx"]:
            raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
        if cfg.resolution <= 0:
            raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")
        
        # Generate grid coordinates
        x_coords = np.arange(-cfg.size[0] / 2, cfg.size[0] / 2 + 1e-9, cfg.resolution)
        y_coords = np.arange(-cfg.size[1] / 2, cfg.size[1] / 2 + 1e-9, cfg.resolution)
        
        if cfg.ordering == "xy":
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='xy')
        else:  # "yx"
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Flatten and create ray starts
        n_rays = grid_x.size
        ray_starts = np.zeros((n_rays, 3), dtype=np.float32)
        ray_starts[:, 0] = grid_x.flatten()
        ray_starts[:, 1] = grid_y.flatten()
        
        # Set ray directions (all parallel)
        ray_directions = np.zeros_like(ray_starts)
        ray_directions[:, :] = np.array(cfg.direction, dtype=np.float32)
        
        # For grid pattern, we need to reshape to [1, n_rays, 3] to match expected format
        # This treats the grid as a single "scan line" with multiple points
        return ray_directions.reshape(1, n_rays, 3)


class LidarPatternGenerator(PatternGenerator):
    """LiDAR pattern generator using Taichi."""
    
    def generate_pattern(self, cfg: LidarPatternCfg) -> np.ndarray:
        """Generate LiDAR pattern."""
        # Vertical angles
        vertical_angles = np.linspace(cfg.vertical_fov_range[0], cfg.vertical_fov_range[1], cfg.channels)
        
        # Handle 360-degree horizontal FOV (exclude last point to avoid overlap)
        h_range = cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]
        if abs(abs(h_range) - 360.0) < 1e-6:
            up_to = -1
        else:
            up_to = None
        
        # Horizontal angles
        num_horizontal_angles = math.ceil(h_range / cfg.horizontal_res)
        horizontal_angles = np.linspace(
            cfg.horizontal_fov_range[0], cfg.horizontal_fov_range[1], num_horizontal_angles
        )[:up_to]
        
        # Convert to radians
        v_rad = np.deg2rad(vertical_angles)
        h_rad = np.deg2rad(horizontal_angles)
        
        # Create meshgrid
        v_angles, h_angles = np.meshgrid(v_rad, h_rad, indexing='ij')
        
        # Spherical to Cartesian conversion (Z is up)
        x = np.cos(v_angles) * np.cos(h_angles)
        y = np.cos(v_angles) * np.sin(h_angles)
        z = np.sin(v_angles)
        
        # Stack and reshape to [n_scan_lines, n_points_per_line, 3]
        ray_directions = np.stack([x, y, z], axis=-1).astype(np.float32)
        
        return ray_directions


class BpearlPatternGenerator(PatternGenerator):
    """Bpearl pattern generator using Taichi."""
    
    def generate_pattern(self, cfg: BpearlPatternCfg) -> np.ndarray:
        """Generate Bpearl pattern."""
        # Horizontal angles
        h_angles = np.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res)
        
        # Vertical angles (predefined for Bpearl)
        v_angles = np.array(cfg.vertical_ray_angles, dtype=np.float32)
        
        # Create meshgrid
        pitch, yaw = np.meshgrid(v_angles, h_angles, indexing='xy')
        pitch_rad = np.deg2rad(pitch.flatten()) + np.pi / 2
        yaw_rad = np.deg2rad(yaw.flatten())
        
        # Spherical to Cartesian
        x = np.sin(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(pitch_rad) * np.sin(yaw_rad)
        z = np.cos(pitch_rad)
        
        # Bpearl uses negative direction convention
        ray_directions = -np.stack([x, y, z], axis=1).astype(np.float32)
        
        # Reshape to [n_scan_lines, n_points_per_line, 3]
        n_v = len(v_angles)
        n_h = len(h_angles)
        ray_directions = ray_directions.reshape(n_v, n_h, 3)
        
        return ray_directions


class SphericalPatternGenerator(PatternGenerator):
    """Spherical uniform pattern generator using Taichi."""
    
    def generate_pattern(self, cfg: SphericalPatternCfg) -> np.ndarray:
        """Generate spherical uniform pattern."""
        # Create angular grids
        vertical_angles = np.linspace(-cfg.fov_vertical / 2, cfg.fov_vertical / 2, cfg.n_scan_lines)
        horizontal_angles = np.linspace(-cfg.fov_horizontal / 2, cfg.fov_horizontal / 2, cfg.n_points_per_line)
        
        # Generate ray vectors in spherical coordinates
        ray_vectors = np.zeros((cfg.n_scan_lines, cfg.n_points_per_line, 3), dtype=np.float32)
        
        for i, v_angle in enumerate(vertical_angles):
            for j, h_angle in enumerate(horizontal_angles):
                v_rad = np.deg2rad(v_angle)
                h_rad = np.deg2rad(h_angle)
                
                # Convert spherical to cartesian (x=forward, y=left, z=up)
                ray_vectors[i, j, 0] = np.cos(v_rad) * np.cos(h_rad)  # x (forward)
                ray_vectors[i, j, 1] = np.cos(v_rad) * np.sin(h_rad)  # y (left)
                ray_vectors[i, j, 2] = np.sin(v_rad)  # z (up)
        
        return ray_vectors


class LivoxPatternGenerator(PatternGenerator):
    """Livox LiDAR pattern generator with caching (prefers precomputed .npy scan patterns)."""
    
    # Livox sensor parameters
    LIVOX_PARAMS = {
        "avia": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 77.2,
            "samples": 24000
        },
        "HAP": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 45300,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
        },
        "horizon": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid40": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid70": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 70.4,
            "samples": 10000,
        },
        "mid360": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 360.0,
            "vertical_fov": 59.0,
            "samples": 20000,
        },
        "tele": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 14.5,
            "vertical_fov": 16.1,
            "samples": 24000,
        }
    }
    
    # Class-level cache for generated patterns
    _pattern_cache: Dict[str, np.ndarray] = {}
    
    def __init__(self):
        super().__init__()
        self.current_start_index = 0
        self.generated_patterns = {}  # Instance-level pattern storage
        # Track last update tick for time-based dynamic updates
        self._last_update_tick: Optional[int] = None
    
    def generate_pattern(self, cfg: LivoxPatternCfg) -> np.ndarray:
        """Generate Livox pattern with caching."""
        if cfg.use_simple_grid:
            return self._generate_simple_grid_pattern(cfg)
        else:
            return self._generate_livox_scan_pattern(cfg)
    
    def _generate_simple_grid_pattern(self, cfg: LivoxPatternCfg) -> np.ndarray:
        """Generate simple grid pattern for Livox sensor."""
        # Convert FOV to radians
        h_fov_min = math.radians(cfg.horizontal_fov_deg_min)
        h_fov_max = math.radians(cfg.horizontal_fov_deg_max)
        v_fov_min = math.radians(cfg.vertical_fov_deg_min)
        v_fov_max = math.radians(cfg.vertical_fov_deg_max)
        
        # Generate grid pattern
        ray_directions = np.zeros((cfg.vertical_line_num, cfg.horizontal_line_num, 3), dtype=np.float32)
        
        for i in range(cfg.vertical_line_num):
            for j in range(cfg.horizontal_line_num):
                # Calculate angles
                if cfg.vertical_line_num > 1:
                    v_angle = v_fov_min + (v_fov_max - v_fov_min) * i / (cfg.vertical_line_num - 1)
                else:
                    v_angle = (v_fov_min + v_fov_max) / 2
                
                if cfg.horizontal_line_num > 1:
                    h_angle = h_fov_min + (h_fov_max - h_fov_min) * j / (cfg.horizontal_line_num - 1)
                else:
                    h_angle = (h_fov_min + h_fov_max) / 2
                
                # Convert to Cartesian (x=forward, y=left, z=up)
                cos_h = math.cos(h_angle)
                sin_h = math.sin(h_angle)
                cos_v = math.cos(v_angle)
                sin_v = math.sin(v_angle)
                
                ray_directions[i, j, 0] = cos_h * cos_v  # x (forward)
                ray_directions[i, j, 1] = sin_h * cos_v  # y (left)
                ray_directions[i, j, 2] = sin_v          # z (up)
        
        return ray_directions
    
    def _generate_livox_scan_pattern(self, cfg: LivoxPatternCfg) -> np.ndarray:
        """Generate realistic Livox scan pattern using NumPy RNG."""
        if cfg.sensor_type not in self.LIVOX_PARAMS:
            raise ValueError(f"Unsupported Livox sensor type: {cfg.sensor_type}")
        
        params = self.LIVOX_PARAMS[cfg.sensor_type]
        
        # Create cache key
        cache_key = self._create_cache_key(cfg, params)
        
        # Check if pattern is already cached
        if cache_key in self._pattern_cache:
            full_pattern = self._pattern_cache[cache_key]
        else:
            # Generate new pattern using Taichi
            full_pattern = self._generate_taichi_pattern(cfg, params)
            self._pattern_cache[cache_key] = full_pattern
        
        # Store pattern for this instance
        self.generated_patterns[cfg.sensor_type] = full_pattern
        
        # Return sampled pattern (first frame)
        return self._sample_pattern(full_pattern, cfg)
    
    def _create_cache_key(self, cfg: LivoxPatternCfg, params: Dict) -> str:
        """Create a unique cache key for the pattern configuration."""
        key_data = {
            'sensor_type': cfg.sensor_type,
            'horizontal_fov': params.get('horizontal_fov', 360.0),
            'vertical_fov': params.get('vertical_fov', 90.0),
            'total_samples': params['samples'] * 10,  # Generate enough for temporal sampling
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_taichi_pattern(self, cfg: LivoxPatternCfg, params: Dict) -> np.ndarray:
        """Load Livox pattern angles from precomputed .npy files when available.
        Falls back to NumPy RNG when files are missing.
        Returns array of shape (N, 2) with columns [theta, phi] in radians.
        """
        import os
        # Map sensor type to pattern filename (note HAP is upper-case)
        pattern_files = {
            "avia": "avia.npy",
            "horizon": "horizon.npy",
            "HAP": "HAP.npy",
            "mid360": "mid360.npy",
            "mid40": "mid40.npy",
            "mid70": "mid70.npy",
            "tele": "tele.npy",
        }
        pattern_file = pattern_files.get(cfg.sensor_type)
        pattern_angles: Optional[np.ndarray] = None
        if pattern_file is not None:
            # Local scan_patterns directory relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(script_dir, "patterns", pattern_file)
            pattern_path = local_path
            if not os.path.exists(pattern_path):
                # Optional unified path fallback (kept for compatibility, may not exist)
                omniperc_root = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.dirname(script_dir)
                                )
                            )
                        )
                    )
                )
                unified_dir = os.path.join(
                    omniperc_root,
                    "LidarSensor",
                    "LidarSensor",
                    "sensor_pattern",
                    "sensor_lidar",
                    "scan_mode",
                )
                unified_path = os.path.join(unified_dir, pattern_file)
                if os.path.exists(unified_path):
                    pattern_path = unified_path
            if os.path.exists(pattern_path):
                data = np.load(pattern_path)
                # Expect shape (N, 2): [theta, phi]
                if isinstance(data, np.lib.npyio.NpzFile):
                    # If accidentally using .npz, try common keys
                    if "angles" in data:
                        data = data["angles"]
                    elif "theta" in data and "phi" in data:
                        data = np.stack([data["theta"], data["phi"]], axis=-1)
                    else:
                        # Fallback: try first 2 columns of the first array
                        first_key = list(data.files)[0]
                        data = data[first_key]
                if data.ndim == 2 and data.shape[1] >= 2:
                    pattern_angles = data[:, :2].astype(np.float32)
        # Fallback to RNG if files missing or invalid
        if pattern_angles is None:
            total_samples = params["samples"] * 10
            h_fov = math.radians(params.get("horizontal_fov", 360.0))
            v_fov = math.radians(params.get("vertical_fov", 90.0))
            rng = np.random.default_rng(seed=abs(hash(cfg.sensor_type)) % (2**32))
            pattern_angles = np.empty((total_samples, 2), dtype=np.float32)
            pattern_angles[:, 0] = rng.uniform(-0.5 * h_fov, 0.5 * h_fov, size=total_samples)  # theta
            pattern_angles[:, 1] = rng.uniform(-0.5 * v_fov, 0.5 * v_fov, size=total_samples)  # phi
        return pattern_angles
    
    def _sample_pattern(self, full_pattern: np.ndarray, cfg: LivoxPatternCfg, start_index: Optional[int] = None) -> np.ndarray:
        """Sample a subset of rays from the full pattern.
        If start_index is provided, sampling starts from there; otherwise uses cfg.rolling_window_start.
        """
        total_rays = full_pattern.shape[0]
        samples = min(cfg.samples, total_rays)
        
        # Rolling window sampling start
        if start_index is None:
            start_idx = cfg.rolling_window_start % total_rays
        else:
            start_idx = start_index % total_rays
        
        if start_idx + samples <= total_rays:
            selected_angles = full_pattern[start_idx:start_idx + samples]
        else:
            # Wraparound case
            end_samples = total_rays - start_idx
            begin_samples = samples - end_samples
            selected_angles = np.vstack([
                full_pattern[start_idx:],
                full_pattern[:begin_samples]
            ])
        
        # Apply downsampling if requested
        if cfg.downsample > 1:
            selected_angles = selected_angles[::cfg.downsample]
        
        # Convert angles to Cartesian coordinates
        theta = selected_angles[:, 0]  # horizontal angles
        phi = selected_angles[:, 1]    # vertical angles
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Convert to ray directions (x=forward, y=left, z=up)
        x = cos_theta * cos_phi  # forward component
        y = sin_theta * cos_phi  # left component
        z = sin_phi              # up component
        
        ray_directions = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # Normalize directions
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / norms
        
        # Return as flat array for compatibility with grid patterns
        return ray_directions.reshape(1, -1, 3)
    
    def update_dynamic_pattern(self, cfg: LivoxPatternCfg, time_step: float) -> Optional[np.ndarray]:
        """Update pattern for dynamic Livox sensors by advancing an internal rolling index.
        time_step is treated as the current simulation time in seconds. We update once per tick
        determined by pattern_update_rate (updates per second). No new cfg is created.
        """
        if not cfg.enable_dynamic_pattern or cfg.sensor_type not in self.generated_patterns:
            return None
        
        # Determine whether to update this call. Default: 10 updates/sec (every 0.1s).
        # You can map pattern_rotation_speed to a rate if desired; keep simple for now.
        pattern_update_rate = 10  # Hz
        current_tick = int(time_step * pattern_update_rate + 1e-6)
        if self._last_update_tick is not None and current_tick == self._last_update_tick:
            return None  # Not time to update yet
        
        # Time to update
        self._last_update_tick = current_tick
        full_pattern = self.generated_patterns[cfg.sensor_type]
        total_rays = full_pattern.shape[0]
        
        # Advance rolling window by one frame worth of samples
        self.current_start_index = (self.current_start_index + cfg.samples) % total_rays
        
        # Sample using the updated start index
        return self._sample_pattern(full_pattern, cfg, start_index=self.current_start_index)
    

class SpinningLidarPatternGenerator(PatternGenerator):
    """Spinning lidar pattern generator (Velodyne HDL64/VLP32, Ouster OS128)."""

    VLP32_ANGLES_DEG = np.array([
        -25.0, -22.5, -20.0, -15.0, -13.0, -10.0, -5.0, -3.0,
        -2.333, -1.0, -0.667, -0.333, 0.0, 0.0, 0.333, 0.667,
        1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333,
        3.667, 4.0, 5.0, 7.0, 10.0, 15.0, 17.0, 20.0
    ], dtype=np.float32)

    def generate_pattern(self, cfg: SpinningLidarPatternCfg) -> np.ndarray:
        sensor = cfg.sensor_type.lower()
        if sensor not in {"hdl64", "vlp32", "os128"}:
            raise ValueError(f"Unsupported spinning lidar type: {cfg.sensor_type}")

        # Determine vertical angles (phi) and channel count
        if sensor == "hdl64":
            n_channels = cfg.n_channels if cfg.n_channels is not None else 64
            phi_min, phi_max = np.deg2rad(cfg.phi_fov)
            phi = np.linspace(phi_min, phi_max, n_channels, dtype=np.float32)
            f_rot = cfg.f_rot
            sample_rate = cfg.sample_rate if cfg.sample_rate is not None else 2.2e6
        elif sensor == "vlp32":
            phi = np.deg2rad(self.VLP32_ANGLES_DEG)
            n_channels = phi.shape[0]
            f_rot = cfg.f_rot
            sample_rate = cfg.sample_rate if cfg.sample_rate is not None else 1.2e6
        else:  # os128
            n_channels = cfg.n_channels if cfg.n_channels is not None else 128
            phi = np.deg2rad(np.linspace(-22.5, 22.5, n_channels, dtype=np.float32))
            f_rot = cfg.f_rot if cfg.f_rot is not None else 20.0
            sample_rate = cfg.sample_rate if cfg.sample_rate is not None else 5.2e6

        # Time sequence over one rotation
        t = np.arange(0.0, 1.0 / f_rot, n_channels / sample_rate, dtype=np.float32)[:, None]
        # Horizontal angles (theta)
        theta = (2.0 * np.pi * f_rot * t) % (2.0 * np.pi)

        # Broadcast to grids
        theta_grid = theta + np.zeros((1, n_channels), dtype=np.float32)
        phi_grid = np.zeros_like(theta, dtype=np.float32) + phi

        # Flatten
        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)

        # Convert to directions (x=forward, y=left, z=up)
        cos_theta = np.cos(theta_flat)
        sin_theta = np.sin(theta_flat)
        cos_phi = np.cos(phi_flat)
        sin_phi = np.sin(phi_flat)
        x = cos_theta * cos_phi
        y = sin_theta * cos_phi
        z = sin_phi
        dirs = np.stack([x, y, z], axis=1).astype(np.float32)
        # Normalize (safety)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)
        # Return as [1, N, 3]
        return dirs.reshape(1, -1, 3)


# Pattern generator factory
PATTERN_GENERATORS = {
    GridPatternCfg: GridPatternGenerator,
    LidarPatternCfg: LidarPatternGenerator,
    BpearlPatternCfg: BpearlPatternGenerator,
    SphericalPatternCfg: SphericalPatternGenerator,
    LivoxPatternCfg: LivoxPatternGenerator,
    SpinningLidarPatternCfg: SpinningLidarPatternGenerator,
}


def create_pattern_generator(cfg: PatternBaseCfg) -> PatternGenerator:
    """Create pattern generator based on configuration type."""
    generator_class = PATTERN_GENERATORS.get(type(cfg))
    if generator_class is None:
        raise ValueError(f"Unsupported pattern configuration type: {type(cfg)}")
    return generator_class()


def generate_ray_pattern(cfg: PatternBaseCfg) -> np.ndarray:
    """Generate ray pattern based on configuration.
    
    Args:
        cfg: Pattern configuration
        
    Returns:
        Ray vectors array. Format depends on pattern type:
        - Grid: [n_rays, 3]
        - LiDAR/Bpearl/Spherical: [n_scan_lines, n_points_per_line, 3]
    """
    generator = create_pattern_generator(cfg)
    return generator.generate_pattern(cfg)


def generate_ray_pattern_with_starts(cfg: PatternBaseCfg) -> Tuple[np.ndarray, np.ndarray]:
    """Generate local ray starts and directions.
    
    - For GridPatternCfg, returns true grid starts and parallel directions.
    - For others, returns zero starts with generated directions.
    
    Returns:
        (ray_starts_local [S,P,3], ray_dirs_local [S,P,3])
    """
    if isinstance(cfg, GridPatternCfg):
        # Recompute grid like GridPatternGenerator but also return starts
        if cfg.ordering not in ["xy", "yx"]:
            raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
        if cfg.resolution <= 0:
            raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")
        x_coords = np.arange(-cfg.size[0] / 2, cfg.size[0] / 2 + 1e-9, cfg.resolution)
        y_coords = np.arange(-cfg.size[1] / 2, cfg.size[1] / 2 + 1e-9, cfg.resolution)
        if cfg.ordering == "xy":
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='xy')
        else:
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        n_rays = grid_x.size
        starts = np.zeros((1, n_rays, 3), dtype=np.float32)
        starts[0, :, 0] = grid_x.flatten()
        starts[0, :, 1] = grid_y.flatten()
        dirs = np.zeros_like(starts)
        dirs[0, :, :] = np.array(cfg.direction, dtype=np.float32)
        return starts, dirs
    else:
        dirs = generate_ray_pattern(cfg)
        starts = np.zeros_like(dirs, dtype=np.float32)
        return starts, dirs


# Convenience functions for common patterns
def create_spherical_pattern(
    n_scan_lines: int = 32,
    n_points_per_line: int = 64, 
    fov_vertical: float = 30.0,
    fov_horizontal: float = 360.0
) -> np.ndarray:
    """Create spherical uniform pattern (compatible with current LidarSensor)."""
    cfg = SphericalPatternCfg(
        n_scan_lines=n_scan_lines,
        n_points_per_line=n_points_per_line,
        fov_vertical=fov_vertical,
        fov_horizontal=fov_horizontal
    )
    return generate_ray_pattern(cfg)


def create_lidar_pattern(
    channels: int = 32,
    vertical_fov_range: Tuple[float, float] = (-15.0, 15.0),
    horizontal_fov_range: Tuple[float, float] = (-180.0, 180.0),
    horizontal_res: float = 1.0
) -> np.ndarray:
    """Create realistic LiDAR pattern."""
    cfg = LidarPatternCfg(
        channels=channels,
        vertical_fov_range=vertical_fov_range,
        horizontal_fov_range=horizontal_fov_range,
        horizontal_res=horizontal_res
    )
    return generate_ray_pattern(cfg)


def create_bpearl_pattern(
    horizontal_fov: float = 360.0,
    horizontal_res: float = 10.0
) -> np.ndarray:
    """Create Bpearl LiDAR pattern."""
    cfg = BpearlPatternCfg(
        horizontal_fov=horizontal_fov,
        horizontal_res=horizontal_res
    )
    return generate_ray_pattern(cfg)


def create_grid_pattern(
    resolution: float = 0.1,
    size: Tuple[float, float] = (2.0, 2.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    ordering: str = "xy"
) -> np.ndarray:
    """Create grid pattern."""
    cfg = GridPatternCfg(
        resolution=resolution,
        size=size,
        direction=direction,
        ordering=ordering
    )
    return generate_ray_pattern(cfg)


def create_livox_pattern(
    sensor_type: str = "avia",
    samples: int = 24000,
    downsample: int = 1,
    use_simple_grid: bool = False,
    enable_dynamic_pattern: bool = True
) -> np.ndarray:
    """Create Livox LiDAR pattern."""
    cfg = LivoxPatternCfg(
        sensor_type=sensor_type,
        samples=samples,
        downsample=downsample,
        use_simple_grid=use_simple_grid,
        enable_dynamic_pattern=enable_dynamic_pattern
    )
    return generate_ray_pattern(cfg)


def create_spinning_lidar_pattern(
    sensor_type: str = "hdl64",
    f_rot: float = 10.0,
    sample_rate: Optional[float] = None,
    n_channels: Optional[int] = None,
    phi_fov: Tuple[float, float] = (-24.9, 2.0),
) -> np.ndarray:
    """Create pattern for traditional spinning lidars.
    - hdl64 defaults: f_rot=10Hz, sample_rate=2.2e6, n_channels=64, phi_fov as given
    - vlp32 defaults: f_rot=10Hz, sample_rate=1.2e6
    - os128 defaults: f_rot=20Hz, sample_rate=5.2e6, n_channels=128
    """
    # Fill sensible defaults per model when None
    st = sensor_type.lower()
    if st == "vlp32":
        if sample_rate is None:
            sample_rate = 1.2e6
        if n_channels is None:
            n_channels = 32
    elif st == "os128":
        if f_rot is None:
            f_rot = 20.0
        if sample_rate is None:
            sample_rate = 5.2e6
        if n_channels is None:
            n_channels = 128
    else:  # hdl64
        if sample_rate is None:
            sample_rate = 2.2e6
        if n_channels is None:
            n_channels = 64

    cfg = SpinningLidarPatternCfg(
        sensor_type=st,
        f_rot=f_rot,
        sample_rate=sample_rate,
        n_channels=n_channels,
        phi_fov=phi_fov,
    )
    return generate_ray_pattern(cfg)
