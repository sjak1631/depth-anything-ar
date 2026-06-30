"""Depth-aware gravity / collision for the AR cube (numpy only).

When enabled (toggled with SPACE in the demo), the cube falls under gravity.
Collision uses the monocular scene depth: the cube rests when a real surface
is found at (or nearer than) the cube's own depth directly beneath it. If no
supporting surface exists at the cube's depth it keeps falling, with the image
bottom acting as a floor of last resort.

The module only calls ``renderer.render`` so it has no OpenCV/torch dependency
and can be unit-tested headless.
"""

from __future__ import annotations

import numpy as np


class Physics:
    def __init__(
        self,
        gravity: float = 3.5,
        restitution: float = 0.4,
        margin: float = 0.05,
        contact_frac: float = 0.15,
        min_contact_px: int = 12,
        band_frac: float = 0.15,
        sleep_speed: float = 0.25,
        max_dt: float = 0.05,
    ):
        self.gravity = gravity
        self.restitution = restitution
        self.margin = margin
        self.contact_frac = contact_frac
        self.min_contact_px = min_contact_px
        self.band_frac = band_frac
        self.sleep_speed = sleep_speed
        self.max_dt = max_dt
        self.enabled = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def reset(self) -> None:
        self.enabled = False

    # ------------------------------------------------------------------ #

    def step(self, obj, renderer, scene_close, dt):
        """Advance one physics step; returns RenderBuffers for display."""
        if not self.enabled:
            obj.vy = 0.0
            obj.on_ground = False
            return renderer.render(obj)

        dt = float(np.clip(dt, 1e-4, self.max_dt))

        # Resting: stay put until the support disappears (or it's disturbed).
        if obj.on_ground:
            buf = renderer.render(obj)
            if self._supported(buf, scene_close):
                return buf
            obj.on_ground = False  # support gone -> resume falling

        prev_ty = obj.ty
        obj.vy -= self.gravity * dt   # gravity pulls toward -Y (screen down)
        obj.ty += obj.vy * dt
        buf = renderer.render(obj)

        # Floor of last resort: the bottom of the image.
        if self._below_screen(buf, renderer):
            obj.ty = prev_ty
            self._resolve_landing(obj)
            return renderer.render(obj)

        # Depth collision with the real scene.
        if self._supported(buf, scene_close):
            obj.ty = prev_ty   # back out of penetration to the contact pose
            self._resolve_landing(obj)
            return renderer.render(obj)

        return buf

    # ------------------------------------------------------------------ #

    def _resolve_landing(self, obj) -> None:
        if abs(obj.vy) < self.sleep_speed:
            obj.vy = 0.0
            obj.on_ground = True
        else:
            obj.vy = -obj.vy * self.restitution  # bounce

    def _below_screen(self, buf, renderer) -> bool:
        if not buf.mask.any():
            return False
        bottom_row = np.where(buf.mask.any(axis=1))[0].max()
        return bottom_row >= renderer.height - 1

    def _supported(self, buf, scene_close) -> bool:
        """True if a real surface lies at/nearer than the cube's bottom band."""
        rows = np.where(buf.mask.any(axis=1))[0]
        if rows.size == 0:
            return False
        top, bot = int(rows.min()), int(rows.max())
        band = max(2, int(self.band_frac * (bot - top + 1)))
        contact = buf.mask.copy()
        contact[: bot - band + 1] = False
        n = int(contact.sum())
        if n < self.min_contact_px:
            return False
        support = scene_close[contact] >= (buf.close[contact] - self.margin)
        return float(support.mean()) >= self.contact_frac
