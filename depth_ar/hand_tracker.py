"""MediaPipe hand tracking + pinch detection (optional dependency).

Isolated here so the rest of the package imports without MediaPipe installed.
``process`` returns a :class:`HandState` (pinch midpoint in pixels, a pinch
flag with hysteresis, and the 21 landmarks for drawing) or ``None`` when no
hand is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandState:
    point: np.ndarray        # (2,) pixel of the thumb-index midpoint
    pinching: bool
    landmarks_px: np.ndarray  # (21, 2) pixel landmark coordinates


class HandTracker:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8

    def __init__(
        self,
        max_hands: int = 1,
        det_conf: float = 0.6,
        track_conf: float = 0.5,
        pinch_on: float = 0.45,
        pinch_off: float = 0.7,
    ):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self._pinching = False

    def process(self, frame_bgr) -> "HandState | None":
        import cv2

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            self._pinching = False
            return None

        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)

        thumb, index = pts[self.THUMB_TIP], pts[self.INDEX_TIP]
        # Normalize the pinch gap by hand size so it is scale invariant.
        ref = np.linalg.norm(pts[self.WRIST] - pts[self.INDEX_MCP]) + 1e-6
        gap = float(np.linalg.norm(thumb - index) / ref)

        # Hysteresis: easier to keep a pinch than to start one.
        if self._pinching:
            self._pinching = gap < self.pinch_off
        else:
            self._pinching = gap < self.pinch_on

        mid = (thumb + index) / 2.0
        return HandState(point=mid, pinching=self._pinching, landmarks_px=pts)

    def close(self) -> None:
        self._hands.close()
