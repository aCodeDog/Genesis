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
    """Livox LiDAR pattern generator with caching (NumPy-based)."""
    
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
        """Generate Livox pattern using NumPy RNG (no Taichi kernel needed)."""
        total_samples = params['samples'] * 10
        h_fov = math.radians(params.get('horizontal_fov', 360.0))
        v_fov = math.radians(params.get('vertical_fov', 90.0))
        rng = np.random.default_rng(seed=abs(hash(cfg.sensor_type)) % (2**32))
        pattern_angles = np.empty((total_samples, 2), dtype=np.float32)
        pattern_angles[:, 0] = rng.uniform(-0.5 * h_fov, 0.5 * h_fov, size=total_samples)  # theta
        pattern_angles[:, 1] = rng.uniform(-0.5 * v_fov, 0.5 * v_fov, size=total_samples)  # phi
        return pattern_angles
    
    def _sample_pattern(self, full_pattern: np.ndarray, cfg: LivoxPatternCfg) -> np.ndarray:
        """Sample a subset of rays from the full pattern."""
        total_rays = full_pattern.shape[0]
        samples = min(cfg.samples, total_rays)
        
        # Rolling window sampling (like original implementation)
        start_idx = cfg.rolling_window_start % total_rays
        
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
        """Update pattern for dynamic Livox sensors."""
        if not cfg.enable_dynamic_pattern or cfg.sensor_type not in self.generated_patterns:
            return None
        
        # Update rolling window start index based on time
        pattern_update_rate = 100  # Update every 100 time steps
        if int(time_step * pattern_update_rate) != int((time_step - 0.02) * pattern_update_rate):
            # Create new config with updated rolling window
            new_cfg = LivoxPatternCfg(
                sensor_type=cfg.sensor_type,
                samples=cfg.samples,
                downsample=cfg.downsample,
                rolling_window_start=(cfg.rolling_window_start + cfg.samples) % (cfg.samples * 10),
                enable_dynamic_pattern=cfg.enable_dynamic_pattern,
                pattern_rotation_speed=cfg.pattern_rotation_speed
            )
            
            # Generate new sample
            full_pattern = self.generated_patterns[cfg.sensor_type]
            return self._sample_pattern(full_pattern, new_cfg)
        
        return None


# Pattern generator factory
PATTERN_GENERATORS = {
    GridPatternCfg: GridPatternGenerator,
    LidarPatternCfg: LidarPatternGenerator,
    BpearlPatternCfg: BpearlPatternGenerator,
    SphericalPatternCfg: SphericalPatternGenerator,
    LivoxPatternCfg: LivoxPatternGenerator,
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
