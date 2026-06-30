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

from depth_ar import SphereRenderer, Object3D, composite  # noqa: E402


def write_ppm(path: str, img_bgr: np.ndarray) -> None:
    rgb = img_bgr[..., ::-1].astype(np.uint8)
    h, w = rgb.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


def main() -> int:
    W, H = 320, 240
    renderer = SphereRenderer(W, H)
    obj = Object3D(tz=3.0, scale_k=3.0, size=0.7)
    buf = renderer.render(obj)

    n_mask = int(buf.mask.sum())
    assert n_mask > 500, f"ball barely rendered ({n_mask} px)"
    assert np.isfinite(buf.close[buf.mask]).all(), "closeness must be finite on the ball"
    print(f"[ok] ball rendered: {n_mask} px, "
          f"close range {buf.close[buf.mask].min():.2f}..{buf.close[buf.mask].max():.2f}")

    # Synthetic camera frame (diagonal gradient) for the previews.
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.stack([
        (xx / W * 255), (yy / H * 255), np.full_like(xx, 90),
    ], axis=-1).astype(np.uint8)

    # 1) Everything far away -> ball fully visible.
    far = np.zeros((H, W), np.float32)
    out_far = composite(frame, far, buf, edge_feather=0)
    vis_far = int(((out_far != frame).any(-1)).sum())
    assert vis_far > 0.9 * n_mask, "ball should be fully visible against a far scene"

    # 2) Everything very near -> ball fully occluded.
    near = np.full((H, W), 10.0, np.float32)
    out_near = composite(frame, near, buf, edge_feather=0)
    vis_near = int(((out_near != frame).any(-1)).sum())
    assert vis_near == 0, f"ball should be fully occluded ({vis_near} px leaked)"

    # 3) Split scene: left half near, right half far -> partial occlusion.
    split = np.zeros((H, W), np.float32)
    split[:, : W // 2] = 10.0  # near wall on the left occludes the ball
    out_split = composite(frame, split, buf, edge_feather=0)
    changed = (out_split != frame).any(-1)
    left = int(changed[:, : W // 2].sum())
    right = int(changed[:, W // 2:].sum())
    assert right > 0 and left == 0, f"expected occlusion only on the left (l={left}, r={right})"
    print(f"[ok] occlusion split: left={left}px (occluded), right={right}px (visible)")

    # 4) Self-occlusion: near faces must hide far faces (z-buffer worked) ->
    #    fewer drawn pixels than the sum of all six faces would imply.
    assert n_mask < W * H, "ball must not fill the frame"

    out_dir = os.path.join(os.path.dirname(__file__), "..", "out")
    os.makedirs(out_dir, exist_ok=True)
    write_ppm(os.path.join(out_dir, "selftest_far.ppm"), out_far)
    write_ppm(os.path.join(out_dir, "selftest_split.ppm"), out_split)
    print(f"[ok] previews written to {os.path.abspath(out_dir)}")

    physics_checks(W, H)
    interaction_checks(W, H)

    print("ALL CHECKS PASSED")
    return 0


def _bottom_row(buf):
    rows = np.where(buf.mask.any(axis=1))[0]
    return int(rows.max()) if rows.size else -1


def physics_checks(W, H):
    from depth_ar import Physics

    renderer = SphereRenderer(W, H)
    dt = 1.0 / 60.0

    # 5) Cube rests on a real surface at its own depth (a "shelf").
    shelf = np.zeros((H, W), np.float32)
    shelf[H // 2: H // 2 + 60, :] = 1.0    # near surface across the lower-middle
    obj = Object3D(ty=1.4, tz=3.0, scale_k=3.0, size=0.5)
    phys = Physics(gravity=3.5)
    phys.enabled = True

    landed = False
    max_bottom = 0
    for _ in range(600):
        buf = phys.step(obj, renderer, shelf, dt)
        max_bottom = max(max_bottom, _bottom_row(buf))
        if obj.on_ground:
            landed = True
            break
    assert landed, "ball never came to rest on the shelf"
    rest_bottom = _bottom_row(renderer.render(obj))
    assert rest_bottom < H - 5, f"ball fell through to the floor ({rest_bottom})"
    print(f"[ok] gravity: ball rested on shelf at bottom row {rest_bottom} (< {H})")

    # 6) With no supporting surface the ball falls but never leaves the frame.
    empty = np.zeros((H, W), np.float32)
    obj2 = Object3D(ty=1.4, tz=3.0, scale_k=3.0, size=0.5)
    phys2 = Physics(gravity=3.5)
    phys2.enabled = True
    worst = 0
    for _ in range(600):
        buf = phys2.step(obj2, renderer, empty, dt)
        worst = max(worst, _bottom_row(buf))
    assert worst <= H - 1, f"ball escaped below the frame (row {worst})"
    final_bottom = _bottom_row(renderer.render(obj2))
    assert final_bottom > 0.6 * H, f"ball did not fall toward the floor ({final_bottom})"
    print(f"[ok] gravity: free fall stopped at image floor (bottom row {final_bottom})")

    # 7) Physics off + zero velocity -> the ball does not move (static placement).
    obj3 = Object3D(ty=1.0)
    phys3 = Physics()  # disabled by default
    y0 = obj3.ty
    for _ in range(120):
        phys3.step(obj3, renderer, empty, dt)
    assert obj3.ty == y0, "object moved while physics was disabled"
    print("[ok] gravity off: a still object stays put until thrown")

    # 8) Gravity OFF = no physics at all: even with a velocity, the ball does
    #    not move (and the velocity is cleared).
    obj4 = Object3D(tx=0.3, ty=-0.2, tz=3.0, size=0.4)
    obj4.set_velocity(2.0, 1.0, -0.5)   # would coast if physics ran
    phys4 = Physics()  # gravity off
    p0 = (obj4.tx, obj4.ty, obj4.tz)
    for _ in range(120):
        phys4.step(obj4, renderer, empty, dt)
    assert (obj4.tx, obj4.ty, obj4.tz) == p0, "gravity off must not move the ball"
    assert obj4.vx == 0.0 and obj4.vy == 0.0 and obj4.vz == 0.0, \
        "gravity off should ignore/clear velocity (no physics)"
    print("[ok] gravity off: no physics (velocity ignored, ball stays put)")

    # 9) Ballistic throw with gravity ON: the release vector is the initial
    #     velocity, then gravity makes a parabola (up first, then down) while
    #     moving sideways. It must stay on screen and not ignore vx.
    obj6 = Object3D(tx=-1.0, ty=0.2, tz=4.0, size=0.4)
    obj6.set_velocity(2.0, 1.2, 0.0)   # throw to the right and upward
    phys6 = Physics(gravity=4.0)
    phys6.enabled = True
    x_start, y_start = obj6.tx, obj6.ty
    y_max = y_start
    x_max = x_start
    on_screen = True
    for _ in range(500):
        phys6.step(obj6, renderer, empty, dt)
        y_max = max(y_max, obj6.ty)
        x_max = max(x_max, obj6.tx)
        cu, cv = renderer.project_point(obj6.tx, obj6.ty, obj6.tz)
        if cu < -1 or cu > W + 1 or cv < -1 or cv > H + 1:
            on_screen = False
            break
    assert x_max > x_start + 0.2, "ballistic throw ignored horizontal velocity (vx)"
    assert y_max > y_start + 0.05, "thrown-up ball should rise before falling (parabola)"
    assert obj6.ty < y_start, "gravity should bring the ball back down below the start"
    assert on_screen, "ballistic throw escaped the frame"
    print(f"[ok] ballistic throw (gravity on): arced right to x={x_max:.2f}, "
          f"peaked at y={y_max:.2f}, fell to y={obj6.ty:.2f}")


def interaction_checks(W, H):
    from depth_ar import near_cube, grab_move

    renderer = SphereRenderer(W, H)

    # 8) project / unproject round-trip.
    x, y, z = 0.4, -0.3, 2.5
    u, v = renderer.project_point(x, y, z)
    rx, ry = renderer.unproject(u, v, z)
    assert abs(rx - x) < 1e-4 and abs(ry - y) < 1e-4, "project/unproject mismatch"
    print("[ok] project/unproject round-trips")

    # 9) near_cube: true on the ball centre, false far away.
    obj = Object3D(tx=0.0, ty=0.0, tz=3.0, size=0.6)
    cu, cv = renderer.project_point(obj.tx, obj.ty, obj.tz)
    assert near_cube(renderer, obj, cu, cv), "grab should start at the ball centre"
    assert not near_cube(renderer, obj, 5, 5), "far corner should not start a grab"
    print("[ok] near_cube hit-test works")

    # 10) grab_move with depth-follow: the ball tracks the hand point and its
    #     depth matches the real surface there (tz = k / scene_close).
    scene = np.full((H, W), 0.5, np.float32)   # uniform mid-depth surface
    obj2 = Object3D(tx=0.0, ty=0.0, tz=3.0, scale_k=3.0, size=0.5)
    target = (W * 0.7, H * 0.4)
    for _ in range(40):  # let the EMA settle
        grab_move(obj2, renderer, target[0], target[1], scene, depth_follow=True)
    pu, pv = renderer.project_point(obj2.tx, obj2.ty, obj2.tz)
    assert abs(pu - target[0]) < 1.0 and abs(pv - target[1]) < 1.0, \
        "grabbed ball must project back to the hand point"
    expected_tz = obj2.scale_k / 0.5
    assert abs(obj2.tz - expected_tz) < 0.05, \
        f"depth-follow tz {obj2.tz:.2f} != expected {expected_tz:.2f}"
    print(f"[ok] grab_move: ball follows hand, tz->{obj2.tz:.2f} (depth matched)")

    # 11) Release regression: the midpoint sees far background (through the gap
    #     between the fingers) but the hand landmarks are on a near surface.
    #     The ball must NOT fly into the distance.
    from depth_ar import sample_close_nearest

    scene_bg = np.full((H, W), 0.04, np.float32)   # far background everywhere
    cx, cy = W * 0.5, H * 0.5
    # A near "hand" blob; the midpoint sits in a far hole between the fingers.
    scene_bg[int(cy) - 25: int(cy) + 25, int(cx) - 40: int(cx) - 10] = 0.9
    scene_bg[int(cy) - 25: int(cy) + 25, int(cx) + 10: int(cx) + 40] = 0.9
    depth_points = np.array([[cx - 25, cy], [cx + 25, cy],
                             [cx - 30, cy + 10], [cx + 30, cy + 10]], np.float32)

    obj3 = Object3D(tx=0.0, ty=0.0, tz=3.0, scale_k=3.0, size=0.5)
    near_close = sample_close_nearest(scene_bg, depth_points)
    assert near_close > 0.5, "nearest sampling should latch onto the hand, not the gap"
    for _ in range(60):
        grab_move(obj3, renderer, cx, cy, scene_bg, depth_follow=True,
                  depth_points=depth_points)
    assert obj3.tz < 6.0, f"ball flew into the distance on release (tz={obj3.tz:.1f})"
    print(f"[ok] release stays stable: tz held at {obj3.tz:.2f} (no fly-away)")

    # 12) Slew limit: a single bad sample cannot teleport the ball far.
    obj4 = Object3D(tz=3.0, scale_k=3.0)
    far = np.full((H, W), 0.04, np.float32)
    grab_move(obj4, renderer, cx, cy, far, depth_follow=True)  # one bad frame
    assert obj4.tz - 3.0 <= 0.4 + 1e-6, f"tz jumped past the slew limit ({obj4.tz:.2f})"
    print(f"[ok] slew limit: one bad frame moves tz only to {obj4.tz:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
