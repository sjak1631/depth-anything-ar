"""Solid 3D sphere renderer (numpy only, no OpenCV/torch dependency).

The sphere is ray-traced analytically with a pinhole camera: for every pixel in
the sphere's bounding box we intersect the camera ray with the sphere and keep
the front hit. That gives an exact per-pixel camera-space depth ``Z`` and surface
normal, so occlusion against the scene is perspective-correct and shading is
smooth. Each covered pixel stores a *closeness* value ``k / Z`` in the same space
as the normalized scene depth, which the compositor and physics use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_LIGHT = np.array([0.4, 0.6, -1.0], dtype=np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)
_BASE_COLOR = np.array([80, 150, 240], dtype=np.float32)   # BGR (warm orange)


@dataclass
class RenderBuffers:
    """Output of one render pass."""

    color: np.ndarray   # (H, W, 3) float32 BGR
    close: np.ndarray   # (H, W) float32, "closeness" (k / Z), -inf where empty
    mask: np.ndarray    # (H, W) bool, True where the sphere was drawn


class SphereRenderer:
    """Projects and ray-traces a single sphere into pixel buffers."""

    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.resize(width, height, fov_deg)

    def resize(self, width: int, height: int, fov_deg: float = 60.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.f = 0.5 * self.width / math.tan(math.radians(fov_deg) / 2.0)

    def project_point(self, x: float, y: float, z: float):
        """Camera-space point -> pixel (u, v). +Y is up, so v is flipped."""
        z = max(float(z), 1e-3)
        u = self.cx + self.f * x / z
        v = self.cy - self.f * y / z
        return u, v

    def unproject(self, u: float, v: float, z: float):
        """Pixel (u, v) at depth z -> camera-space (x, y)."""
        x = (u - self.cx) * z / self.f
        y = -(v - self.cy) * z / self.f
        return x, y

    def render(self, obj) -> RenderBuffers:
        """Render ``obj`` (centre tx,ty,tz; radius = obj.size)."""
        H, W = self.height, self.width
        color = np.zeros((H, W, 3), dtype=np.float32)
        close = np.full((H, W), -np.inf, dtype=np.float32)

        C = np.array([obj.tx, obj.ty, obj.tz], dtype=np.float32)
        R = float(obj.size)
        Zc = C[2]
        if Zc <= R + 1e-3:           # camera inside / behind the sphere
            return RenderBuffers(color, close, np.zeros((H, W), bool))

        # Conservative on-screen radius (silhouette), plus a margin.
        r_px = self.f * R / math.sqrt(max(Zc * Zc - R * R, 1e-6))
        cu, cv = self.project_point(C[0], C[1], C[2])
        minx = max(int(math.floor(cu - r_px)) - 1, 0)
        maxx = min(int(math.ceil(cu + r_px)) + 1, W - 1)
        miny = max(int(math.floor(cv - r_px)) - 1, 0)
        maxy = min(int(math.ceil(cv + r_px)) + 1, H - 1)
        if maxx < minx or maxy < miny:
            return RenderBuffers(color, close, np.zeros((H, W), bool))

        ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
        # Camera ray directions d = ((u-cx)/f, -(v-cy)/f, 1).
        dx = (xs - self.cx) / self.f
        dy = -(ys - self.cy) / self.f
        dz = np.ones_like(dx)

        a = dx * dx + dy * dy + dz * dz
        dC = dx * C[0] + dy * C[1] + dz * C[2]
        c = float(C @ C - R * R)
        disc = dC * dC - a * c
        hit = disc >= 0.0
        if not hit.any():
            return RenderBuffers(color, close, np.zeros((H, W), bool))

        sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
        t = (dC - sqrt_disc) / a          # front intersection (smaller root)
        hit &= t > 1e-3
        if not hit.any():
            return RenderBuffers(color, close, np.zeros((H, W), bool))

        Z = t                              # point.z = t * dz, dz = 1
        # Surface normal (camera space) for shading.
        px = t * dx - C[0]
        py = t * dy - C[1]
        pz = t * dz - C[2]
        inv_r = 1.0 / R
        ndl = -(px * _LIGHT[0] + py * _LIGHT[1] + pz * _LIGHT[2]) * inv_r
        shade = 0.3 + 0.7 * np.clip(ndl, 0.0, 1.0)

        region_close = close[miny:maxy + 1, minx:maxx + 1]
        region_color = color[miny:maxy + 1, minx:maxx + 1]
        cube_close = obj.scale_k / np.maximum(Z, 1e-3)
        region_close[hit] = cube_close[hit].astype(np.float32)
        region_color[hit] = (shade[hit][:, None] * _BASE_COLOR[None, :])

        mask = np.isfinite(close)
        return RenderBuffers(color=color, close=close, mask=mask)
