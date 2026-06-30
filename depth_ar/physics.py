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
        linear_damping: float = 0.8,
        tz_near: float = 0.6,
        tz_far: float = 12.0,
        border_margin: int = 4,
        penetrate_frac: float = 0.4,
    ):
        self.gravity = gravity
        self.restitution = restitution
        self.margin = margin
        self.contact_frac = contact_frac
        self.min_contact_px = min_contact_px
        self.band_frac = band_frac
        self.sleep_speed = sleep_speed
        self.max_dt = max_dt
        # Throw (inertia) parameters, used when gravity is off.
        self.linear_damping = linear_damping   # velocity *= (1 - damping*dt)
        self.tz_near = tz_near
        self.tz_far = tz_far
        self.border_margin = border_margin
        self.penetrate_frac = penetrate_frac
        self.enabled = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def reset(self) -> None:
        self.enabled = False

    # ------------------------------------------------------------------ #

    def step(self, obj, renderer, scene_close, dt):
        """Advance one physics step; returns RenderBuffers for display.

        Gravity on -> the cube falls and lands (vertical). Gravity off -> the
        cube coasts with whatever throw velocity it has (3D inertia), so it can
        be thrown by hand; it slows via drag and bounces off the borders / real
        scene until it comes to rest.
        """
        dt = float(np.clip(dt, 1e-4, self.max_dt))
        if self.enabled:
            return self._step_gravity(obj, renderer, scene_close, dt)
        return self._step_inertia(obj, renderer, scene_close, dt)

    def _step_gravity(self, obj, renderer, scene_close, dt):
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

    def _step_inertia(self, obj, renderer, scene_close, dt):
        """Gravity-off throw: coast on velocity, drag, bounce, then sleep."""
        obj.on_ground = False
        speed2 = obj.vx ** 2 + obj.vy ** 2 + obj.vz ** 2
        if speed2 < self.sleep_speed ** 2:
            obj.stop()
            return renderer.render(obj)

        # Air drag.
        damp = max(0.0, 1.0 - self.linear_damping * dt)
        obj.vx *= damp
        obj.vy *= damp
        obj.vz *= damp

        prev = (obj.tx, obj.ty, obj.tz)
        obj.tx += obj.vx * dt
        obj.ty += obj.vy * dt
        obj.tz += obj.vz * dt

        # Depth walls: bounce when flying too near or too far.
        if obj.tz < self.tz_near:
            obj.tz = self.tz_near
            obj.vz = -obj.vz * self.restitution
        elif obj.tz > self.tz_far:
            obj.tz = self.tz_far
            obj.vz = -obj.vz * self.restitution

        buf = renderer.render(obj)

        # Screen-edge bounce (keeps the cube in view).
        if self._bounce_borders(obj, renderer):
            buf = renderer.render(obj)

        # Scene collision: if the cube has run into a real surface (mostly
        # swallowed by nearer geometry), back out and bounce off it.
        if self._penetrating(buf, scene_close):
            obj.tx, obj.ty, obj.tz = prev
            obj.vx *= -self.restitution
            obj.vy *= -self.restitution
            obj.vz *= -self.restitution
            buf = renderer.render(obj)

        return buf

    def _bounce_borders(self, obj, renderer) -> bool:
        """Reflect velocity and clamp the cube centre inside the image."""
        cu, cv = renderer.project_point(obj.tx, obj.ty, obj.tz)
        m = self.border_margin
        z = obj.tz
        hit = False

        if cu < m or cu > renderer.width - 1 - m:
            x_lo, _ = renderer.unproject(m, cv, z)
            x_hi, _ = renderer.unproject(renderer.width - 1 - m, cv, z)
            obj.tx = float(np.clip(obj.tx, min(x_lo, x_hi), max(x_lo, x_hi)))
            obj.vx = -obj.vx * self.restitution
            hit = True

        if cv < m or cv > renderer.height - 1 - m:
            _, y_lo = renderer.unproject(cu, m, z)
            _, y_hi = renderer.unproject(cu, renderer.height - 1 - m, z)
            obj.ty = float(np.clip(obj.ty, min(y_lo, y_hi), max(y_lo, y_hi)))
            obj.vy = -obj.vy * self.restitution
            hit = True

        return hit

    def _penetrating(self, buf, scene_close) -> bool:
        if not buf.mask.any():
            return False
        m = buf.mask
        n = int(m.sum())
        if n < self.min_contact_px:
            return False
        blocked = scene_close[m] > (buf.close[m] + self.margin)
        return float(blocked.mean()) >= self.penetrate_frac

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
