"""Mutable state of the virtual object plus keyboard control mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Object3D:
    """Pose / appearance of the AR cube, in camera space.

    Camera sits at the origin looking down +Z. ``tz`` is the distance in front
    of the camera. ``scale_k`` is the *manual depth scale* that aligns the
    cube's metric ``1/Z`` closeness with the scene's normalized closeness.
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 3.0
    yaw: float = 0.5
    pitch: float = 0.4
    roll: float = 0.0
    size: float = 0.6          # half edge length in world units
    scale_k: float = 3.0       # manual depth scale (k in close = k / Z)

    # Physics state (used when gravity is enabled).
    vy: float = 0.0            # vertical velocity in world units / s
    on_ground: bool = False    # resting on a supporting surface

    # Step sizes.
    move_step: float = 0.08
    depth_step: float = 0.12
    rot_step: float = math.radians(6)
    size_step: float = 0.06
    k_step: float = 0.12

    # Defaults captured for reset.
    _defaults: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._defaults:
            self._defaults = dict(
                tx=self.tx, ty=self.ty, tz=self.tz,
                yaw=self.yaw, pitch=self.pitch, roll=self.roll,
                size=self.size, scale_k=self.scale_k,
            )

    def reset(self) -> None:
        for k, v in self._defaults.items():
            setattr(self, k, v)
        self.vy = 0.0
        self.on_ground = False

    def handle_key(self, key: int) -> bool:
        """Apply a key press (lowercased ASCII code from cv2.waitKey).

        Returns True if the key was consumed.
        """
        if key in (-1, 255):
            return False
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 256 else ""

        # Translate in the image plane (W/A/S/D), depth with Q/E.
        if ch == "w":
            self.ty += self.move_step
        elif ch == "s":
            self.ty -= self.move_step
        elif ch == "a":
            self.tx -= self.move_step
        elif ch == "d":
            self.tx += self.move_step
        elif ch == "q":
            self.tz = max(0.3, self.tz - self.depth_step)   # closer
        elif ch == "e":
            self.tz += self.depth_step                       # farther
        # Rotation: I/K pitch, J/L yaw, U/O roll.
        elif ch == "i":
            self.pitch += self.rot_step
        elif ch == "k":
            self.pitch -= self.rot_step
        elif ch == "j":
            self.yaw -= self.rot_step
        elif ch == "l":
            self.yaw += self.rot_step
        elif ch == "u":
            self.roll -= self.rot_step
        elif ch == "o":
            self.roll += self.rot_step
        # Cube size.
        elif ch in ("+", "="):
            self.size += self.size_step
        elif ch in ("-", "_"):
            self.size = max(0.05, self.size - self.size_step)
        # Manual depth scale k.
        elif ch == "]":
            self.scale_k += self.k_step
        elif ch == "[":
            self.scale_k = max(0.05, self.scale_k - self.k_step)
        elif ch == "r":
            self.reset()
            return True
        else:
            return False
        # Any manual manipulation wakes a resting object so gravity re-applies.
        self.on_ground = False
        return True
