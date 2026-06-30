"""Grab-and-move math for dragging the cube with a hand (numpy only).

The MediaPipe-specific code lives in :mod:`depth_ar.hand_tracker`; this module
contains only the geometry so it can be unit-tested without MediaPipe/OpenCV:

* whether a screen point is on/near the cube (to start a grab),
* sampling the scene depth under a point,
* moving the cube to follow a screen point, optionally matching its depth to
  the real surface there.
"""

from __future__ import annotations

import numpy as np


def sample_close(scene_close: np.ndarray, px: float, py: float, win: int = 7) -> float:
    """Median scene closeness in a small window around a pixel."""
    H, W = scene_close.shape
    x, y = int(round(px)), int(round(py))
    x0, x1 = max(0, x - win), min(W, x + win + 1)
    y0, y1 = max(0, y - win), min(H, y + win + 1)
    patch = scene_close[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def sample_close_nearest(scene_close: np.ndarray, points, win: int = 7) -> float:
    """Closeness of the *nearest* surface among several probe points.

    The hand is the foreground, so taking the maximum closeness over hand
    landmarks rejects the far background that shows through the gaps between the
    fingers as the hand opens (the cause of the ball flying away)."""
    vals = [sample_close(scene_close, p[0], p[1], win) for p in points]
    return max(vals) if vals else 0.0


def near_cube(renderer, obj, px: float, py: float, extra_px: float = 30.0) -> bool:
    """True if (px, py) is close to the cube's projected centre."""
    cu, cv = renderer.project_point(obj.tx, obj.ty, obj.tz)
    # Approximate on-screen radius of the cube plus a grab margin.
    radius = 1.5 * renderer.f * obj.size / max(obj.tz, 1e-3) + extra_px
    return (cu - px) ** 2 + (cv - py) ** 2 <= radius * radius


def grab_move(
    obj,
    renderer,
    px: float,
    py: float,
    scene_close: np.ndarray,
    depth_follow: bool = True,
    smooth: float = 0.5,
    eps: float = 0.12,
    tz_range=(0.3, 15.0),
    depth_points=None,
    max_tz_step: float = 0.4,
) -> None:
    """Move the cube so it tracks the hand point (px, py).

    With ``depth_follow`` the cube's depth is matched to the real surface under
    the hand: ``k / tz = scene_close`` -> ``tz = k / scene_close`` (EMA-smoothed).
    Then x/y are unprojected from the screen point at that depth.

    To avoid the ball flying into the distance when the hand opens, depth is
    sampled from the *nearest* of several hand landmarks (``depth_points``) rather
    than a single point, and the per-frame depth change is slew-limited by
    ``max_tz_step``.
    """
    if depth_follow:
        if depth_points is not None and len(depth_points) > 0:
            close = sample_close_nearest(scene_close, depth_points)
        else:
            close = sample_close(scene_close, px, py)
        close = max(close, eps)
        tz_target = float(np.clip(obj.scale_k / close, tz_range[0], tz_range[1]))
        delta = smooth * (tz_target - obj.tz)
        delta = float(np.clip(delta, -max_tz_step, max_tz_step))  # slew limit
        obj.tz += delta

    obj.tx, obj.ty = renderer.unproject(px, py, obj.tz)
    # Held by the hand: it is not resting on anything. Velocity is tracked by the
    # caller (from the hand motion) so it can be thrown on release.
    obj.on_ground = False
