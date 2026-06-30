"""Depth-Anything based AR demo: occlusion + interactive object placement.

Pipeline (per frame):
    webcam frame -> Depth-Anything V2 Small depth -> normalize to closeness
    -> render 3D cube (perspective) -> compare cube vs scene depth -> composite.

Run:
    python main.py                 # default webcam (index 0)
    python main.py --source 1      # another camera
    python main.py --cpu           # force CPU
    python main.py --depth-interval 2   # run depth every 2 frames (faster)

Controls are printed on screen (press H to toggle the help overlay).
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from depth_ar import (
    SphereRenderer, Object3D, Physics, composite, normalize_depth, metric_closeness,
    depth_to_color, near_cube, grab_move, collide_ball, VelocityTracker,
)

RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

HELP_LINES = [
    "WASD: move   Q/E: closer/farther   +/-: size   [ ]: depth scale k",
    "Close your whole hand over the ball to grab & move it",
    "Not grabbing? Hands & feet HIT the ball and it bounces off (gravity ON)",
    "Gravity OFF: no physics (the ball just stays where you leave it)",
    "SPACE: gravity on/off   V: depth   H: help   R: reset   ESC: quit",
]


def parse_args():
    p = argparse.ArgumentParser(description="Depth-Anything AR (occlusion + placement)")
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--metric", action="store_true",
                   help="metric mode: real meters via Depth-Anything-V2-Metric-Indoor-Small")
    p.add_argument("--model", default=None, help="override the depth model id")
    p.add_argument("--metric-min", type=float, default=0.2, help="metric: clamp near depth (m)")
    p.add_argument("--metric-max", type=float, default=10.0, help="metric: clamp far depth (m)")
    p.add_argument("--fov", type=float, default=60.0,
                   help="camera horizontal field of view (deg) used for rendering")
    p.add_argument("--infer-size", type=int, default=392, help="depth inference long side (px)")
    p.add_argument("--depth-interval", type=int, default=1, help="run depth every N frames")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--no-fp16", action="store_true", help="disable half precision on GPU")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the webcam horizontally")
    p.add_argument("--invert-depth", action="store_true",
                   help="(relative mode) set if model outputs larger=farther")
    p.add_argument("--bias", type=float, default=0.0, help="occlusion bias (-1..1)")
    p.add_argument("--feather", type=int, default=1, help="edge feather radius in px (0=off)")
    p.add_argument("--gravity", type=float, default=None,
                   help="gravity (units/s^2); default 9.8 metric, 3.5 relative")
    p.add_argument("--restitution", type=float, default=None,
                   help="bounciness 0..1; default 0.5 metric, 0.4 relative")
    p.add_argument("--no-hands", action="store_true", help="disable MediaPipe hand grabbing")
    p.add_argument("--max-hands", type=int, default=2, help="max hands to track")
    p.add_argument("--grasp-on", type=float, default=1.0,
                   help="grasp (fist) start threshold; smaller = must close more")
    p.add_argument("--grasp-off", type=float, default=1.3, help="grasp release threshold")
    p.add_argument("--no-depth-follow", action="store_true",
                   help="while grabbing, keep tz fixed instead of matching the hand's depth")
    p.add_argument("--throw-window", type=float, default=0.15,
                   help="seconds of velocity history averaged into the throw speed")
    p.add_argument("--no-feet", action="store_true",
                   help="disable MediaPipe pose / foot hitting")
    p.add_argument("--hit-restitution", type=float, default=0.9,
                   help="bounciness when a hand/foot hits the ball (0..1+)")
    p.add_argument("--hit-speed", type=float, default=1.2,
                   help="min impact speed (world units/s) to register a hit")
    p.add_argument("--hit-gain", type=float, default=1.0,
                   help="scale applied to the post-hit ball speed (>1 = livelier)")
    p.add_argument("--hit-reach", type=float, default=55.0,
                   help="hand/foot-to-ball reach in px to count as a touch")
    return p.parse_args()


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    # Prefer MJPEG to avoid raw YUYV being misread as BGR (causes green frames).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hand(img, hand, grabbed):
    """Draw landmarks and the palm point (colour shows grasp/grab state)."""
    for (x, y) in hand.landmarks_px:
        cv2.circle(img, (int(x), int(y)), 2, (200, 200, 200), -1, cv2.LINE_AA)
    px, py = int(hand.point[0]), int(hand.point[1])
    if grabbed:
        color = (0, 255, 0)        # green: holding the ball
    elif hand.grasping:
        color = (0, 200, 255)      # orange: closed fist but not on the ball
    else:
        color = (160, 160, 160)    # grey: open hand
    cv2.circle(img, (px, py), 12, color, 2, cv2.LINE_AA)


def draw_foot(img, foot, hot):
    """Draw the toe point; bright when it just hit the ball."""
    px, py = int(foot.point[0]), int(foot.point[1])
    color = (0, 255, 255) if hot else (120, 120, 120)
    cv2.circle(img, (px, py), 9, color, 2, cv2.LINE_AA)
    cv2.putText(img, foot.label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def draw_overlay(img, obj, fps, show_help, physics, metric):
    phys = "ON" if physics.enabled else "off"
    if physics.enabled and obj.on_ground:
        phys = "RESTING"
    if metric:
        lines = [
            f"fps {fps:4.1f}  [metric]  tz {obj.tz:4.2f}m  "
            f"r {obj.size:4.2f}m  gravity {phys}",
        ]
    else:
        lines = [
            f"fps {fps:4.1f}  tz {obj.tz:4.2f}  k {obj.scale_k:4.2f}  "
            f"size {obj.size:4.2f}  gravity {phys}",
        ]
    if show_help:
        lines += HELP_LINES
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main():
    args = parse_args()

    # Lazy import so --help works without torch installed.
    from depth_ar.depth_estimator import DepthEstimator

    cap = open_capture(args.source, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first frame from the source.")
    H, W = frame.shape[:2]

    model_name = args.model or (METRIC_MODEL if args.metric else RELATIVE_MODEL)
    device = "cpu" if args.cpu else None
    print(f"Loading {model_name} ({'metric' if args.metric else 'relative'}) ...")
    estimator = DepthEstimator(
        model_name=model_name,
        device=device,
        fp16=not args.no_fp16,
        infer_size=args.infer_size,
    )
    print(f"Depth model ready on: {estimator.device}")

    renderer = SphereRenderer(W, H, fov_deg=args.fov)
    gravity = args.gravity if args.gravity is not None else (9.8 if args.metric else 3.5)
    restitution = args.restitution if args.restitution is not None else (0.5 if args.metric else 0.4)
    if args.metric:
        # Metric mode: everything is in real meters; closeness = 1/Z so scale_k=1.
        obj = Object3D(tz=1.5, size=0.12, scale_k=1.0,
                       move_step=0.05, depth_step=0.1, size_step=0.02)
        physics = Physics(gravity=gravity, restitution=restitution,
                          sleep_speed=0.05, tz_near=0.3, tz_far=8.0)
    else:
        obj = Object3D()
        physics = Physics(gravity=gravity, restitution=restitution)

    tracker = None
    if not args.no_hands:
        try:
            from depth_ar.hand_tracker import HandTracker
            tracker = HandTracker(
                max_hands=args.max_hands,
                grasp_on=args.grasp_on,
                grasp_off=args.grasp_off,
            )
            print("Hand tracking enabled (close your hand over the ball to grab).")
        except Exception as exc:  # MediaPipe missing or failed to init.
            print(f"Hand tracking disabled ({exc}). Install mediapipe to enable.")

    pose = None
    if not args.no_feet:
        try:
            from depth_ar.pose_tracker import PoseTracker
            pose = PoseTracker()
            print("Foot tracking enabled (hit the ball with your foot; gravity ON).")
        except Exception as exc:
            print(f"Foot tracking disabled ({exc}). Install mediapipe to enable.")

    scene_close = np.zeros((H, W), dtype=np.float32)
    show_help = True
    show_depth = False
    grabbed = False
    grab_label = None           # which hand (Left/Right) is holding the ball
    prev_grab_pos = None        # last ball pos while grabbed (for throw velocity)
    throw_vel = VelocityTracker(window=args.throw_window)
    prev_hitters = {}           # hitter id -> last pixel pos (for hit velocity)
    frame_idx = 0
    last = time.time()
    fps = 0.0

    win = "Depth-Anything AR"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))

        now = time.time()
        dt = now - last
        last = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        if frame_idx % max(1, args.depth_interval) == 0:
            depth = estimator.infer(frame)
            if args.metric:
                scene_close = metric_closeness(depth, args.metric_min, args.metric_max)
            else:
                scene_close = normalize_depth(depth, invert=args.invert_depth)

        # Hand interaction: close your whole hand over the ball to grab it.
        hands = tracker.process(frame) if tracker is not None else []

        active = None
        if grabbed:
            # Continuity: keep the SAME hand as long as it is present and still
            # closed. This is what makes two-handed scenes stable (no flicker
            # between hands).
            for h in hands:
                if h.label == grab_label and h.grasping:
                    active = h
                    break
            if active is None:
                # Released: throw with the moving-average velocity over the last
                # throw-window seconds (robust to lag / the hand slowing as it
                # opens), so the ball leaves the hand with natural inertia.
                obj.set_velocity(*throw_vel.average())
                obj.on_ground = False
                grabbed = False
                grab_label = None
                prev_grab_pos = None

        if not grabbed:
            # Start a grab with the closed hand nearest the ball.
            cu, cv = renderer.project_point(obj.tx, obj.ty, obj.tz)
            best_d = None
            for h in hands:
                if not h.grasping:
                    continue
                if not near_cube(renderer, obj, h.point[0], h.point[1]):
                    continue
                d = (cu - h.point[0]) ** 2 + (cv - h.point[1]) ** 2
                if best_d is None or d < best_d:
                    best_d, active = d, h
            if active is not None:
                grabbed = True
                grab_label = active.label
                prev_grab_pos = None
                throw_vel.reset()

        if grabbed and active is not None:
            # Probe depth from points on the palm/fingers (never the background)
            # so releasing can't fling the ball away.
            lm = active.landmarks_px
            depth_points = lm[[0, 5, 9, 13, 17, 8, 12]]
            grab_move(obj, renderer, active.point[0], active.point[1], scene_close,
                      depth_follow=not args.no_depth_follow,
                      depth_points=depth_points)
            # Continuously record the ball's velocity while held; the throw uses
            # the moving average of this history at release.
            cur = np.array([obj.tx, obj.ty, obj.tz], dtype=np.float64)
            if prev_grab_pos is not None:
                throw_vel.add(now, (cur - prev_grab_pos) / max(dt, 1e-3))
            prev_grab_pos = cur

        # Collision interaction: when NOT holding the ball, hands and feet act as
        # moving colliders — touch the ball and it bounces off. A hit imparts
        # velocity, so it only moves the ball when gravity is ON.
        feet = pose.process(frame) if pose is not None else []
        hit_ids = set()
        hitters = []
        if not grabbed:
            for h in hands:               # palm of every hand
                hitters.append((f"hand:{h.label}", h.point))
            for foot in feet:             # toe of every foot
                hitters.append((f"foot:{foot.label}", foot.point))

        seen_ids = set()
        for hid, cur in hitters:
            seen_ids.add(hid)
            prev = prev_hitters.get(hid)
            prev_hitters[hid] = np.asarray(cur, dtype=np.float64)
            if prev is None or not physics.enabled:
                continue
            if collide_ball(obj, renderer, prev, cur, dt, scene_close,
                            restitution=args.hit_restitution, reach_px=args.hit_reach,
                            min_speed=args.hit_speed, gain=args.hit_gain):
                hit_ids.add(hid)
        # Keep history only for hitters still tracked this frame.
        prev_hitters = {k: prev_hitters[k] for k in seen_ids}

        if grabbed:
            # Held by the hand: position is set directly, physics paused.
            buffers = renderer.render(obj)
        else:
            # Gravity on -> ballistic motion; gravity off -> stays put.
            buffers = physics.step(obj, renderer, scene_close, dt)
        out = composite(frame, scene_close, buffers,
                        bias=args.bias, edge_feather=args.feather)

        for h in hands:
            draw_hand(out, h, grabbed and h.label == grab_label)
        for foot in feet:
            draw_foot(out, foot, f"foot:{foot.label}" in hit_ids)

        if show_depth:
            dvis = depth_to_color(scene_close)
            out = cv2.addWeighted(out, 0.6, dvis, 0.4, 0)

        draw_overlay(out, obj, fps, show_help, physics, args.metric)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("h"):
            show_help = not show_help
        elif key == ord("v"):
            show_depth = not show_depth
        elif key == ord(" "):
            physics.toggle()
        elif key == ord("r"):
            obj.handle_key(key)
            physics.reset()
        else:
            obj.handle_key(key)

        frame_idx += 1

    cap.release()
    if tracker is not None:
        tracker.close()
    if pose is not None:
        pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
