"""Solid 3D cube renderer (numpy only, no OpenCV/torch dependency).

The cube is rasterized with a perspective camera (pinhole model). For every
covered pixel we compute a *closeness* value ``k / Z`` (Z = camera-space depth),
which lives in the same space as the normalized scene depth produced by
:mod:`depth_ar.compositor`. That lets the compositor decide, per pixel, whether
the cube is in front of (visible) or behind (occluded by) the real scene.

Inverse depth (``1/Z``) is interpolated across each triangle because it is
linear in screen space, giving perspective-correct occlusion for free.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Cube vertices centered on the origin, edge half-length = 1.
_VERTS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)

# Each face: (4 vertex indices, outward normal, base BGR colour).
_FACES = [
    ((0, 1, 2, 3), (0, 0, -1), (60, 60, 230)),   # -Z  red
    ((4, 5, 6, 7), (0, 0, 1), (60, 230, 60)),    # +Z  green
    ((0, 3, 7, 4), (-1, 0, 0), (230, 120, 40)),  # -X  blue
    ((1, 2, 6, 5), (1, 0, 0), (40, 200, 230)),   # +X  yellow
    ((0, 1, 5, 4), (0, -1, 0), (230, 60, 200)),  # -Y  magenta
    ((3, 2, 6, 7), (0, 1, 0), (60, 200, 230)),   # +Y  orange
]

_LIGHT = np.array([0.4, 0.6, -1.0], dtype=np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)


@dataclass
class RenderBuffers:
    """Output of one cube render pass."""

    color: np.ndarray   # (H, W, 3) float32 BGR
    close: np.ndarray   # (H, W) float32, cube "closeness" (k / Z), -inf where empty
    mask: np.ndarray    # (H, W) bool, True where the cube was drawn


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return ry @ rx @ rz


class CubeRenderer:
    """Projects and rasterizes a single cube into pixel buffers."""

    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.resize(width, height, fov_deg)

    def resize(self, width: int, height: int, fov_deg: float = 60.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        # Focal length in pixels from horizontal field of view.
        self.f = 0.5 * self.width / math.tan(math.radians(fov_deg) / 2.0)

    def render(self, obj) -> RenderBuffers:
        """Render ``obj`` (a :class:`depth_ar.object3d.Object3D`)."""
        H, W = self.height, self.width
        color = np.zeros((H, W, 3), dtype=np.float32)
        close = np.full((H, W), -np.inf, dtype=np.float32)

        rot = _rotation_matrix(obj.yaw, obj.pitch, obj.roll)
        translate = np.array([obj.tx, obj.ty, obj.tz], dtype=np.float32)

        # Camera-space vertices.
        verts_cam = (_VERTS * obj.size) @ rot.T + translate  # (8, 3)
        Z = verts_cam[:, 2]
        safe_z = np.maximum(Z, 1e-3)
        inv_z = 1.0 / safe_z
        # Project to screen. +X right, +Y up (so flip v).
        u = self.cx + self.f * verts_cam[:, 0] * inv_z
        v = self.cy - self.f * verts_cam[:, 1] * inv_z
        screen = np.stack([u, v], axis=1)  # (8, 2)

        for idx, normal, base in _FACES:
            # Skip faces that have any vertex behind / too close to the camera.
            if np.any(Z[list(idx)] <= 1e-2):
                continue
            n_cam = rot @ np.asarray(normal, dtype=np.float32)
            shade = 0.35 + 0.65 * max(0.0, float(-np.dot(n_cam, _LIGHT)))
            col = np.clip(np.asarray(base, dtype=np.float32) * shade, 0, 255)
            # Two triangles per quad.
            for a, b, c in ((idx[0], idx[1], idx[2]), (idx[0], idx[2], idx[3])):
                self._raster_triangle(
                    screen[a], screen[b], screen[c],
                    inv_z[a], inv_z[b], inv_z[c],
                    col, obj.scale_k, color, close,
                )

        mask = np.isfinite(close)
        return RenderBuffers(color=color, close=close, mask=mask)

    def _raster_triangle(self, p0, p1, p2, iz0, iz1, iz2, col, k, color, close):
        H, W = self.height, self.width
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if maxx < minx or maxy < miny:
            return

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            return

        ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)

        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            return

        inv_z = a * iz0 + b * iz1 + c * iz2  # perspective-correct (linear in screen)
        cube_close = k * inv_z

        region_close = close[miny:maxy + 1, minx:maxx + 1]
        win = inside & (cube_close > region_close)
        if not win.any():
            return
        region_close[win] = cube_close[win]
        region_color = color[miny:maxy + 1, minx:maxx + 1]
        region_color[win] = col
