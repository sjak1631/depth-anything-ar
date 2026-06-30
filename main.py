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
    CubeRenderer, Object3D, Physics, composite, normalize_depth, depth_to_color,
)

HELP_LINES = [
    "WASD: move   Q/E: closer/farther   +/-: size",
    "IJKL: rotate(pitch/yaw)   U/O: roll   [ ]: depth scale k",
    "SPACE: gravity on/off   V: depth view   H: help   R: reset   ESC: quit",
]


def parse_args():
    p = argparse.ArgumentParser(description="Depth-Anything AR (occlusion + placement)")
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--infer-size", type=int, default=392, help="depth inference long side (px)")
    p.add_argument("--depth-interval", type=int, default=1, help="run depth every N frames")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--no-fp16", action="store_true", help="disable half precision on GPU")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the webcam horizontally")
    p.add_argument("--invert-depth", action="store_true",
                   help="set if model outputs larger=farther (metric models)")
    p.add_argument("--bias", type=float, default=0.0, help="occlusion bias (-1..1)")
    p.add_argument("--feather", type=int, default=1, help="edge feather radius in px (0=off)")
    p.add_argument("--gravity", type=float, default=3.5, help="gravity strength (world units/s^2)")
    p.add_argument("--restitution", type=float, default=0.4, help="bounciness 0..1 on collision")
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


def draw_overlay(img, obj, fps, show_help, physics):
    phys = "ON" if physics.enabled else "off"
    if physics.enabled and obj.on_ground:
        phys = "RESTING"
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

    device = "cpu" if args.cpu else None
    print(f"Loading {args.model} ...")
    estimator = DepthEstimator(
        model_name=args.model,
        device=device,
        fp16=not args.no_fp16,
        infer_size=args.infer_size,
    )
    print(f"Depth model ready on: {estimator.device}")

    renderer = CubeRenderer(W, H)
    obj = Object3D()
    physics = Physics(gravity=args.gravity, restitution=args.restitution)

    scene_close = np.zeros((H, W), dtype=np.float32)
    show_help = True
    show_depth = False
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
            scene_close = normalize_depth(depth, invert=args.invert_depth)

        # Advance gravity/collision (renders the cube; no-op physics when off).
        buffers = physics.step(obj, renderer, scene_close, dt)
        out = composite(frame, scene_close, buffers,
                        bias=args.bias, edge_feather=args.feather)

        if show_depth:
            dvis = depth_to_color(scene_close)
            out = cv2.addWeighted(out, 0.6, dvis, 0.4, 0)

        draw_overlay(out, obj, fps, show_help, physics)

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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
