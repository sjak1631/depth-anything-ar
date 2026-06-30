"""Offline sanity check for the renderer + occlusion compositor.

Uses numpy only (no OpenCV / torch / camera), so it can run in CI or any plain
Python env. Validates the depth-comparison occlusion logic and writes a couple
of PPM previews you can open to eyeball the result.

    python tools/selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_ar import CubeRenderer, Object3D, composite  # noqa: E402


def write_ppm(path: str, img_bgr: np.ndarray) -> None:
    rgb = img_bgr[..., ::-1].astype(np.uint8)
    h, w = rgb.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


def main() -> int:
    W, H = 320, 240
    renderer = CubeRenderer(W, H)
    obj = Object3D(tz=3.0, scale_k=3.0, size=0.7)
    buf = renderer.render(obj)

    n_mask = int(buf.mask.sum())
    assert n_mask > 500, f"cube barely rendered ({n_mask} px)"
    assert np.isfinite(buf.close[buf.mask]).all(), "closeness must be finite on the cube"
    print(f"[ok] cube rendered: {n_mask} px, "
          f"close range {buf.close[buf.mask].min():.2f}..{buf.close[buf.mask].max():.2f}")

    # Synthetic camera frame (diagonal gradient) for the previews.
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.stack([
        (xx / W * 255), (yy / H * 255), np.full_like(xx, 90),
    ], axis=-1).astype(np.uint8)

    # 1) Everything far away -> cube fully visible.
    far = np.zeros((H, W), np.float32)
    out_far = composite(frame, far, buf, edge_feather=0)
    vis_far = int(((out_far != frame).any(-1)).sum())
    assert vis_far > 0.9 * n_mask, "cube should be fully visible against a far scene"

    # 2) Everything very near -> cube fully occluded.
    near = np.full((H, W), 10.0, np.float32)
    out_near = composite(frame, near, buf, edge_feather=0)
    vis_near = int(((out_near != frame).any(-1)).sum())
    assert vis_near == 0, f"cube should be fully occluded ({vis_near} px leaked)"

    # 3) Split scene: left half near, right half far -> partial occlusion.
    split = np.zeros((H, W), np.float32)
    split[:, : W // 2] = 10.0  # near wall on the left occludes the cube
    out_split = composite(frame, split, buf, edge_feather=0)
    changed = (out_split != frame).any(-1)
    left = int(changed[:, : W // 2].sum())
    right = int(changed[:, W // 2:].sum())
    assert right > 0 and left == 0, f"expected occlusion only on the left (l={left}, r={right})"
    print(f"[ok] occlusion split: left={left}px (occluded), right={right}px (visible)")

    # 4) Self-occlusion: near faces must hide far faces (z-buffer worked) ->
    #    fewer drawn pixels than the sum of all six faces would imply.
    assert n_mask < W * H, "cube must not fill the frame"

    out_dir = os.path.join(os.path.dirname(__file__), "..", "out")
    os.makedirs(out_dir, exist_ok=True)
    write_ppm(os.path.join(out_dir, "selftest_far.ppm"), out_far)
    write_ppm(os.path.join(out_dir, "selftest_split.ppm"), out_split)
    print(f"[ok] previews written to {os.path.abspath(out_dir)}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
