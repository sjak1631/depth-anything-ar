"""Scene-depth normalization and occlusion-aware compositing (numpy only)."""

from __future__ import annotations

import numpy as np


def normalize_depth(
    depth: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    invert: bool = False,
) -> np.ndarray:
    """Map a raw depth/disparity map to a *closeness* image in ``[0, 1]``.

    Depth-Anything V2 (relative) outputs larger values for *nearer* surfaces,
    so by default higher input -> higher closeness. Robust percentile scaling
    keeps the result stable against a few extreme pixels.

    Set ``invert=True`` if your model outputs larger values for *farther*
    surfaces (true metric depth).
    """
    d = depth.astype(np.float32)
    if invert:
        d = -d
    lo, hi = np.percentile(d, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def composite(
    frame: np.ndarray,
    scene_close: np.ndarray,
    buffers,
    bias: float = 0.0,
    edge_feather: int = 1,
) -> np.ndarray:
    """Blend the rendered cube over ``frame`` with depth occlusion.

    A cube pixel is shown only where it is closer than the scene:
    ``cube_close + bias >= scene_close``.

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR camera frame.
    scene_close : (H, W) float32 in [0, 1] from :func:`normalize_depth`.
    buffers : RenderBuffers from :class:`depth_ar.renderer.SphereRenderer`.
    bias : nudge that biases the ball towards (positive) or against (negative)
        being in front; useful to fight depth noise at contact edges.
    edge_feather : optional Gaussian feathering (px) of the visibility mask to
        soften occlusion boundaries. 0 disables it.
    """
    out = frame.astype(np.float32)

    visible = buffers.mask & (buffers.close + bias >= scene_close)
    if not visible.any():
        return frame

    alpha = visible.astype(np.float32)
    if edge_feather and edge_feather > 0:
        # Local import keeps the module importable without OpenCV for tests.
        import cv2

        ksize = edge_feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
        # Never paint where the cube was not drawn at all.
        alpha *= buffers.mask.astype(np.float32)

    alpha3 = alpha[..., None]
    out = out * (1.0 - alpha3) + buffers.color * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_to_color(scene_close: np.ndarray) -> np.ndarray:
    """Colourize a closeness map for a debug overlay (requires OpenCV)."""
    import cv2

    vis = (np.clip(scene_close, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
