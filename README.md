# Depth-Anything AR (occlusion + object placement)

Python + OpenCV demo that turns a fixed PC webcam into a tiny AR scene. Each
frame is sent through **Depth-Anything V2 Small** to estimate depth, a virtual
3D cube is rendered with a perspective camera, and the cube is **occluded by
real objects** in the scene by comparing per-pixel depth. The cube is moved
around with the keyboard.

```
webcam frame
   └─ Depth-Anything V2 Small ─► relative depth ─► normalize to "closeness" [0,1]
   └─ render 3D cube (pinhole) ─► per-pixel closeness  k / Z
        └─ compare cube vs scene closeness ─► occlusion mask ─► composite
```

## How occlusion works

Depth-Anything (relative) gives a depth map where **larger = nearer**. It is
normalized per frame to a *closeness* image in `[0, 1]`. The cube's geometry
produces a camera-space depth `Z` per pixel; its closeness is `k / Z`, where
`k` is a **manual scale** you tune live (`[` / `]`) so the cube's metric depth
lines up with the scene's relative depth. A cube pixel is drawn only where
`cube_close >= scene_close` — i.e. the cube is in front of the real surface.
Inverse depth (`1/Z`) is interpolated across triangles, so occlusion is
perspective-correct, and a z-buffer handles the cube's own self-occlusion.

## Install

```bash
# 1. (recommended) create a venv
python -m venv .venv && source .venv/bin/activate

# 2. install PyTorch for your CUDA version first (https://pytorch.org), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
pip install -r requirements.txt
```

The Depth-Anything weights (~100 MB) download automatically from Hugging Face
on first run.

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
| `W` `A` `S` `D` | move the cube in the image plane (up/left/down/right) |
| `Q` / `E` | move closer / farther (depth `tz`) |
| `I` `K` / `J` `L` | rotate pitch / yaw |
| `U` / `O` | roll |
| `+` / `-` | grow / shrink the cube |
| `[` / `]` | decrease / increase depth scale `k` (occlusion alignment) |
| `SPACE` | toggle gravity — drop the cube; it collides with real objects |
| `V` | toggle depth-map overlay |
| `H` | toggle help overlay |
| `R` | reset the object (also turns gravity off) |
| `ESC` | quit |

**Tuning tip:** if the cube is wrongly occluded (or never occluded), nudge `k`
with `[` / `]` until objects pass in front of / behind it correctly, then move
it with `Q`/`E`.

## Gravity & collision (SPACE)

Press `SPACE` to toggle gravity. While on, the cube falls and **collides with
the real scene using depth**: at each step the depth of the surface directly
beneath the cube's bottom is compared with the cube's own depth. If a real
surface sits at (or nearer than) the cube's depth, the cube lands and rests on
it (with a small bounce, `--restitution`); otherwise it keeps falling, and the
bottom of the image acts as a floor. Moving the cube (`WASD`/`QE`) or `R` wakes
it so it can fall again.

So the cube only lands on things at its own depth — line the cube up with a real
surface (tune `tz` with `Q`/`E` and `k` with `[`/`]`) and it will sit on top of
it. Tune the feel with `--gravity` and `--restitution`.

## Project layout

```
main.py                     # webcam loop, controls, OpenCV display
depth_ar/
  depth_estimator.py        # Depth-Anything V2 (transformers) wrapper
  renderer.py               # solid 3D cube rasterizer (numpy, perspective + z-buffer)
  object3d.py               # object pose/state + keyboard mapping
  compositor.py             # depth normalize + occlusion compositing
  physics.py                # depth-aware gravity + collision (SPACE)
tools/selftest.py           # offline numpy-only check (no camera/torch needed)
```

## Offline self-test

Validates the rendering + occlusion math without a camera, GPU, or torch:

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
