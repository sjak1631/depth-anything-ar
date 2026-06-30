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
    eps: float = 0.05,
    tz_range=(0.3, 50.0),
) -> None:
    """Move the cube so it tracks the hand point (px, py).

    With ``depth_follow`` the cube's depth is matched to the real surface under
    the hand: ``k / tz = scene_close`` -> ``tz = k / scene_close`` (EMA-smoothed).
    Then x/y are unprojected from the screen point at that depth.
    """
    if depth_follow:
        close = max(sample_close(scene_close, px, py), eps)
        tz_target = float(np.clip(obj.scale_k / close, tz_range[0], tz_range[1]))
        obj.tz += smooth * (tz_target - obj.tz)

    obj.tx, obj.ty = renderer.unproject(px, py, obj.tz)
    # Held by the hand: cancel any physics motion.
    obj.vy = 0.0
    obj.on_ground = False
