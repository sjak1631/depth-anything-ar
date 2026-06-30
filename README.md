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
`--bias` (occlusion edge tuning), `--feather` (soften occlusion edges, px).

### Metric mode (real meters)

```bash
python main.py --metric          # Depth-Anything-V2-Metric-Indoor-Small, real meters
```

By default the demo uses the **relative** model and a manual scale `k` to line up
the ball with the scene. With `--metric` it instead uses
**Depth-Anything-V2-Metric-Indoor-Small**, which outputs depth in real meters, so
everything works in physical units:

- Scene depth → closeness as inverse depth `1/Z`. The renderer already produces
  `k/Z`, so with `k = 1` the ball's depth and the scene's depth compare directly
  — **no manual scale needed**.
- The ball lives in meters (`tz` in m, radius in m), gravity is `9.8 m/s²`, and
  throw/hit velocities are in m/s.
- Grabbing snaps the ball to the **real metric depth** under your hand.

Tunables: `--metric-min` / `--metric-max` (clamp the depth range, m), `--fov`
(camera horizontal field of view used for rendering — set it close to your
webcam's for correct on-screen ball size), and the usual `--gravity` /
`--restitution` (default to metric-appropriate values in this mode).

The Metric-Indoor model is trained for indoor scenes; for outdoor use, pass
`--model depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`.

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
  (`--restitution`), loses sideways speed to ground friction, and settles. The
  throw velocity is the **moving average** of the hand's velocity over the last
  `--throw-window` seconds (not the single instant of release), so it stays
  natural despite frame lag or the hand slowing as it opens.
- **Gravity OFF** — **no physics at all.** The ball is only moved directly by
  your hand; release it and it simply stays where you left it (floating in
  place).

Depth-follow is robust against a classic failure (the background showing through
the hand spiking the depth and flinging the ball away): depth is sampled from the
*nearest* of several hand landmarks and the per-frame depth change is slew-limited.

Flags: `--no-hands` (disable), `--max-hands N` (default 2), `--grasp-on` /
`--grasp-off` (how closed the hand must be, normalized by hand size),
`--no-depth-follow` (keep `tz` fixed while dragging).

## Hit & bounce (hands and feet)

When you are **not** holding the ball, your **hands and feet act as moving
colliders** — touch the ball and it bounces off. The hitter is treated as an
infinitely heavy moving paddle: with contact normal `n` (from the hitter to the
ball) and the relative velocity along it, the ball reflects that component
(`v' = v - (1+e)·vₙ·n`). So a resting ball struck by a moving hand shoots away in
the hit direction with the hand's speed, and a ball flying at you is bounced back.
Feet are tracked with MediaPipe Pose so you can kick it too.

A hit only registers when the hitter is at roughly the ball's depth and actually
moving into it (a receding or too-gentle touch adds no energy, which avoids
jitter). Because a hit imparts velocity, it only moves the ball when **gravity is
ON** (gravity off runs no physics).

Flags: `--no-feet` (disable foot tracking), `--hit-restitution` (bounciness),
`--hit-speed` (min impact speed to register), `--hit-gain` (scale post-hit speed;
>1 = livelier), `--hit-reach` (hand/foot-to-ball reach in px).

## Project layout

```
main.py                     # webcam loop, controls, OpenCV display
depth_ar/
  depth_estimator.py        # Depth-Anything V2 (transformers) wrapper
  renderer.py               # analytic ray-traced sphere (numpy, exact per-pixel depth)
  object3d.py               # object pose/velocity state + keyboard mapping
  compositor.py             # depth normalize + occlusion compositing
  physics.py                # gravity-on 3D ballistic sim; gravity-off = no physics
  interaction.py            # grab/move + collision (hit/bounce) geometry (numpy only)
  hand_tracker.py           # MediaPipe hand + whole-hand grasp detection (optional)
  pose_tracker.py           # MediaPipe pose + foot tracking for hitting (optional)
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
- For real-world metric placement (depth in meters, no manual scale), use
  `--metric` (see *Metric mode* above).
