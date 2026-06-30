# Depth-Anything AR (occlusion + object placement)

Python + OpenCV demo that turns a fixed PC webcam into a tiny AR scene. Each
frame is sent through **Depth-Anything V2 Small** to estimate depth, a virtual
3D **ball (sphere)** is rendered with a perspective camera, and the ball is
**occluded by real objects** in the scene by comparing per-pixel depth. The ball
is moved with the keyboard or grabbed with your hand.

```
webcam frame
   └─ Depth-Anything V2 Small ─► relative depth ─► normalize to "closeness" [0,1]
   └─ ray-trace 3D sphere (pinhole) ─► per-pixel closeness  k / Z
        └─ compare ball vs scene closeness ─► occlusion mask ─► composite
```

## How occlusion works

Depth-Anything (relative) gives a depth map where **larger = nearer**. It is
normalized per frame to a *closeness* image in `[0, 1]`. The sphere is ray-traced
analytically, giving an exact camera-space depth `Z` per pixel; its closeness is
`k / Z`, where `k` is a **manual scale** you tune live (`[` / `]`) so the ball's
metric depth lines up with the scene's relative depth. A ball pixel is drawn only
where `ball_close >= scene_close` — i.e. the ball is in front of the real
surface. Because each pixel has its true depth, occlusion (and self-occlusion) is
correct and smoothly shaded.

## Install

```bash
# 1. (recommended) create a venv
python -m venv .venv && source .venv/bin/activate

# 2. install PyTorch for your CUDA version first (https://pytorch.org), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
pip install -r requirements.txt
```

The Depth-Anything weights (~100 MB) and the MediaPipe hand model download
automatically on first run.

## Run

```bash
python main.py                    # default webcam (index 0), CUDA if available
python main.py --source 1         # pick another camera
python main.py --cpu              # force CPU
python main.py --depth-interval 2 # run depth every 2nd frame (more FPS)
python main.py --infer-size 308   # smaller depth input = faster, coarser
```

Useful flags: `--no-fp16` (disable half precision), `--no-mirror`,
`--invert-depth` (for true metric models), `--bias` (occlusion edge tuning),
`--feather` (soften occlusion edges, px).

## Controls

| Keys | Action |
|------|--------|
| `W` `A` `S` `D` | move the ball in the image plane (up/left/down/right) |
| `Q` / `E` | move closer / farther (depth `tz`) |
| `+` / `-` | grow / shrink the ball |
| `[` / `]` | decrease / increase depth scale `k` (occlusion alignment) |
| close hand | close your whole hand over the ball to grab it; move to drag |
| `SPACE` | toggle gravity (on = ballistic physics, off = no physics) |
| `V` | toggle depth-map overlay |
| `H` | toggle help overlay |
| `R` | reset the ball (also turns gravity off) |
| `ESC` | quit |

**Tuning tip:** if the ball is wrongly occluded (or never occluded), nudge `k`
with `[` / `]` until objects pass in front of / behind it correctly, then move
it with `Q`/`E`.

## Grab and throw with your hand (MediaPipe)

With `mediapipe` installed, the webcam tracks your hand(s). **Close your whole
hand into a fist over the ball** to grab it (the palm marker turns green); open
your hand to release. While grabbed, the ball follows your palm, and its depth
follows the real surface under your hand (`tz = k / scene_close`), so it tracks
your hand in 3D and stays consistent with occlusion.

**Two hands:** both hands are tracked and the grab sticks to whichever hand
started it (matched by Left/Right identity), so having a second hand in view
won't make the ball jump between hands.

**Throwing depends on gravity:**

- **Gravity ON** — releasing throws the ball with your hand's velocity as the
  initial vector of a full 3D **ballistic simulation**: it arcs under gravity (no
  air drag, an accurate parabola), bounces off real surfaces and the floor
  (`--restitution`), loses sideways speed to ground friction, and settles.
- **Gravity OFF** — **no physics at all.** The ball is only moved directly by
  your hand; release it and it simply stays where you left it (floating in
  place).

Depth-follow is robust against a classic failure (the background showing through
the hand spiking the depth and flinging the ball away): depth is sampled from the
*nearest* of several hand landmarks and the per-frame depth change is slew-limited.

Flags: `--no-hands` (disable), `--max-hands N` (default 2), `--grasp-on` /
`--grasp-off` (how closed the hand must be, normalized by hand size),
`--no-depth-follow` (keep `tz` fixed while dragging).

## Kick with your foot (MediaPipe Pose)

The webcam also tracks your body, so you can **kick** the ball with your foot.
When a foot (toe) sweeps into the ball fast enough — and is at roughly the ball's
depth — the foot's velocity is transferred to the ball. Because a kick is an
impulse (it sets the ball's velocity), it only produces motion when **gravity is
ON** (gravity off runs no physics): the ball gets launched and arcs ballistically.

The toe marker brightens (yellow) when a foot is in range and moving fast enough
to kick. A short cooldown prevents one swing from registering many times.

Flags: `--no-feet` (disable), `--kick-speed` (min foot speed to register, world
units/s), `--kick-gain` (fraction of foot velocity transferred; >1 = livelier),
`--kick-reach` (foot-to-ball reach in px to count as a touch).

## Project layout

```
main.py                     # webcam loop, controls, OpenCV display
depth_ar/
  depth_estimator.py        # Depth-Anything V2 (transformers) wrapper
  renderer.py               # analytic ray-traced sphere (numpy, exact per-pixel depth)
  object3d.py               # object pose/velocity state + keyboard mapping
  compositor.py             # depth normalize + occlusion compositing
  physics.py                # gravity-on 3D ballistic sim; gravity-off = no physics
  interaction.py            # grab/move/kick geometry (numpy only)
  hand_tracker.py           # MediaPipe hand + whole-hand grasp detection (optional)
  pose_tracker.py           # MediaPipe pose + foot tracking for kicking (optional)
tools/selftest.py           # offline numpy-only check (no camera/torch needed)
```

## Offline self-test

Validates the rendering, occlusion, physics and grab math without a camera, GPU,
or torch:

```bash
python tools/selftest.py     # writes previews to out/*.ppm
```

## Notes / limitations

- Designed for a **fixed** camera. Relative depth is per-frame, so a moving
  camera makes the alignment `k` drift.
- Monocular depth is noisy at thin structures and edges; `--feather` and
  `--bias` help smooth contact boundaries.
- For real-world metric placement, switch to a `Depth-Anything-V2-Metric-*`
  model and pass `--invert-depth`.
