"""
Depth camera (pinhole) ray pattern generation for Genesis.

Follows the same style as lidar_pattern: define a config, a generator, and helpers
that return local-frame ray starts and directions of shape [H, W, 3].

Coordinate convention (robotics camera frame):
- x: forward, y: left, z: up
- Derived from standard camera frame (x right, y down, z forward) via:
  [x_r, y_r, z_r] = [z_c, -x_c, -y_c]
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Type
import math
import numpy as np


# ------------------------- Base -------------------------
@dataclass
class PatternBaseCfg:
    """Base configuration for a pattern."""
    pass


class PatternGenerator:
    """Base class for pattern generators."""

    def generate_pattern(self, cfg: PatternBaseCfg) -> np.ndarray:  # [S,P,3] or [H,W,3]
        raise NotImplementedError


# ------------------------- Depth Camera -------------------------
@dataclass
class DepthCameraPatternCfg(PatternBaseCfg):
    """Pinhole depth camera pattern configuration.

    You can provide intrinsics (fx, fy, cx, cy). If missing, they will be computed
    from image size and FOVs when provided.
    """
    width: int = 640
    height: int = 480
    # Intrinsics (in pixels)
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    # Alternative specification via FOV (degrees)
    fov_horizontal: Optional[float] = 90.0
    fov_vertical: Optional[float] = None


class DepthCameraPatternGenerator(PatternGenerator):
    """Generate ray directions for a pinhole camera."""

    def generate_pattern(self, cfg: DepthCameraPatternCfg) -> np.ndarray:
        W, H = int(cfg.width), int(cfg.height)
        if W <= 0 or H <= 0:
            raise ValueError("width and height must be positive")

        # Derive intrinsics if needed
        fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
        fh, fv = cfg.fov_horizontal, cfg.fov_vertical

        if fx is None or fy is None:
            if fh is None and fv is None:
                # Default FOVs if nothing provided
                fh = 90.0
            if fh is not None and fv is None:
                # preserve aspect ratio
                fh_rad = math.radians(fh)
                fv_rad = 2.0 * math.atan((H / W) * math.tan(fh_rad / 2.0))
            elif fv is not None and fh is None:
                fv_rad = math.radians(fv)
                fh_rad = 2.0 * math.atan((W / H) * math.tan(fv_rad / 2.0))
            else:
                fh_rad = math.radians(fh)
                fv_rad = math.radians(fv)
            fx = W / (2.0 * math.tan(fh_rad / 2.0))
            fy = H / (2.0 * math.tan(fv_rad / 2.0))
        if cx is None:
            cx = W * 0.5
        if cy is None:
            cy = H * 0.5

        # Pixel centers
        u = np.arange(0, W, dtype=np.float32) + 0.5  # shape (W,)
        v = np.arange(0, H, dtype=np.float32) + 0.5  # shape (H,)
        uu, vv = np.meshgrid(u, v, indexing="xy")  # (H, W)

        # Camera frame (x right, y down, z forward)
        x_c = (uu - cx) / float(fx)
        y_c = (vv - cy) / float(fy)
        z_c = np.ones_like(x_c, dtype=np.float32)

        # Robotics camera frame (x forward, y left, z up): [z, -x, -y]
        x_r = z_c
        y_r = -x_c
        z_r = -y_c
        dirs = np.stack([x_r, y_r, z_r], axis=-1).astype(np.float32)  # (H, W, 3)

        # Normalize
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        return dirs  # (H, W, 3)


# ------------------------- Factory + Helpers -------------------------
PATTERN_GENERATORS: Dict[Type[PatternBaseCfg], Type[PatternGenerator]] = {
    DepthCameraPatternCfg: DepthCameraPatternGenerator,
}


def create_pattern_generator(cfg: PatternBaseCfg) -> PatternGenerator:
    gen_cls = PATTERN_GENERATORS.get(type(cfg))
    if gen_cls is None:
        raise ValueError(f"Unsupported pattern configuration type: {type(cfg)}")
    return gen_cls()


def generate_ray_pattern(cfg: PatternBaseCfg) -> np.ndarray:
    generator = create_pattern_generator(cfg)
    return generator.generate_pattern(cfg)


def generate_ray_pattern_with_starts(cfg: PatternBaseCfg) -> Tuple[np.ndarray, np.ndarray]:
    dirs = generate_ray_pattern(cfg)
    starts = np.zeros_like(dirs, dtype=np.float32)
    return starts, dirs


# Convenience

def create_depth_camera_pattern(
    width: int = 640,
    height: int = 480,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    fov_horizontal: Optional[float] = 90.0,
    fov_vertical: Optional[float] = None,
) -> np.ndarray:
    cfg = DepthCameraPatternCfg(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fov_horizontal=fov_horizontal,
        fov_vertical=fov_vertical,
    )
    return generate_ray_pattern(cfg)
