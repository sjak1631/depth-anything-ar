"""Depth-aware physics for the AR cube (numpy only).

Two modes, switched by ``enabled`` (SPACE in the demo):

* **Gravity on** — a full 3D ballistic simulation. The cube is integrated with
  its current velocity (the hand's release vector, when thrown) plus gravity on
  the vertical axis, with *no* air drag, so it follows an accurate parabola. It
  bounces off real surfaces and the image floor (restitution) and loses
  tangential speed to ground friction, eventually coming to rest.
* **Gravity off** — inertial float: the cube coasts on its velocity with air
  drag and bounces around until it sleeps (used for zero-g throws / dragging).

Collision uses the monocular scene depth: a surface "supports" the cube where it
sits at (or nearer than) the cube's own depth beneath it. The module only calls
``renderer.render`` so it has no OpenCV/torch dependency and is headless-testable.
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
        ground_friction: float = 0.6,
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
        self.linear_damping = linear_damping     # air drag (gravity-off float)
        self.ground_friction = ground_friction   # tangential loss on contact
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
        """Advance one physics step; returns RenderBuffers for display."""
        dt = float(np.clip(dt, 1e-4, self.max_dt))
        if self.enabled:
            return self._step_gravity(obj, renderer, scene_close, dt)
        return self._step_inertia(obj, renderer, scene_close, dt)

    def _step_gravity(self, obj, renderer, scene_close, dt):
        """3D ballistic motion: initial velocity + gravity, with bouncing."""
        # Resting on a surface: stay put until the support disappears.
        if obj.on_ground:
            buf = renderer.render(obj)
            if self._supported(buf, scene_close):
                return buf
            obj.on_ground = False  # support gone -> resume falling

        obj.vy -= self.gravity * dt   # ballistic: gravity only, no air drag
        prev = (obj.tx, obj.ty, obj.tz)
        obj.tx += obj.vx * dt
        obj.ty += obj.vy * dt
        obj.tz += obj.vz * dt
        self._bounce_depth(obj)

        buf = renderer.render(obj)

        # Side / top containment (the image bottom is the floor, handled below).
        if self._bounce_borders(obj, renderer, bounce_bottom=False):
            buf = renderer.render(obj)

        # Land on the image floor.
        if self._below_screen(buf, renderer):
            obj.ty = prev[1]
            self._resolve_landing(obj)
            return renderer.render(obj)

        # Land on a real surface at the cube's depth (only when descending).
        if obj.vy <= 0 and self._supported(buf, scene_close):
            obj.ty = prev[1]
            self._resolve_landing(obj)
            return renderer.render(obj)

        # Hit a real surface head-on (sideways / forward) -> bounce off it.
        if self._penetrating(buf, scene_close):
            obj.tx, obj.ty, obj.tz = prev
            obj.vx *= -self.restitution
            obj.vy *= -self.restitution
            obj.vz *= -self.restitution
            return renderer.render(obj)

        return buf

    def _step_inertia(self, obj, renderer, scene_close, dt):
        """Gravity-off throw: coast on velocity, drag, bounce, then sleep."""
        obj.on_ground = False
        speed2 = obj.vx ** 2 + obj.vy ** 2 + obj.vz ** 2
        if speed2 < self.sleep_speed ** 2:
            obj.stop()
            return renderer.render(obj)

        damp = max(0.0, 1.0 - self.linear_damping * dt)   # air drag
        obj.vx *= damp
        obj.vy *= damp
        obj.vz *= damp

        prev = (obj.tx, obj.ty, obj.tz)
        obj.tx += obj.vx * dt
        obj.ty += obj.vy * dt
        obj.tz += obj.vz * dt
        self._bounce_depth(obj)

        buf = renderer.render(obj)
        if self._bounce_borders(obj, renderer, bounce_bottom=True):
            buf = renderer.render(obj)

        if self._penetrating(buf, scene_close):
            obj.tx, obj.ty, obj.tz = prev
            obj.vx *= -self.restitution
            obj.vy *= -self.restitution
            obj.vz *= -self.restitution
            buf = renderer.render(obj)

        return buf

    # ------------------------------------------------------------------ #

    def _resolve_landing(self, obj) -> None:
        """Vertical bounce + tangential friction; sleep when slow enough."""
        if abs(obj.vy) < self.sleep_speed:
            obj.vy = 0.0
        else:
            obj.vy = -obj.vy * self.restitution
        obj.vx *= self.ground_friction
        obj.vz *= self.ground_friction
        if obj.vx ** 2 + obj.vy ** 2 + obj.vz ** 2 < self.sleep_speed ** 2:
            obj.stop()
            obj.on_ground = True

    def _bounce_depth(self, obj) -> None:
        if obj.tz < self.tz_near:
            obj.tz = self.tz_near
            obj.vz = -obj.vz * self.restitution
        elif obj.tz > self.tz_far:
            obj.tz = self.tz_far
            obj.vz = -obj.vz * self.restitution

    def _bounce_borders(self, obj, renderer, bounce_bottom: bool = True) -> bool:
        """Reflect velocity and clamp the cube centre inside the image."""
        cu, cv = renderer.project_point(obj.tx, obj.ty, obj.tz)
        m = self.border_margin
        z = obj.tz
        hit = False

        if cu < m:
            x_left, _ = renderer.unproject(m, cv, z)
            obj.tx = max(obj.tx, x_left)
            obj.vx = -obj.vx * self.restitution
            hit = True
        elif cu > renderer.width - 1 - m:
            x_right, _ = renderer.unproject(renderer.width - 1 - m, cv, z)
            obj.tx = min(obj.tx, x_right)
            obj.vx = -obj.vx * self.restitution
            hit = True

        if cv < m:  # top edge (+Y)
            _, y_top = renderer.unproject(cu, m, z)
            obj.ty = min(obj.ty, y_top)
            obj.vy = -obj.vy * self.restitution
            hit = True
        elif bounce_bottom and cv > renderer.height - 1 - m:  # bottom edge (-Y)
            _, y_bot = renderer.unproject(cu, renderer.height - 1 - m, z)
            obj.ty = max(obj.ty, y_bot)
            obj.vy = -obj.vy * self.restitution
            hit = True

        return hit

    def _penetrating(self, buf, scene_close) -> bool:
        """True if the cube has run into a real surface (mostly occluded)."""
        if not buf.mask.any():
            return False
        m = buf.mask
        if int(m.sum()) < self.min_contact_px:
            return False
        blocked = scene_close[m] > (buf.close[m] + self.margin)
        return float(blocked.mean()) >= self.penetrate_frac

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
