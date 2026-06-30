"""MediaPipe hand tracking + pinch detection (MediaPipe Tasks API, >=0.10).

Isolated here so the rest of the package imports without MediaPipe installed.
``process`` returns a :class:`HandState` (pinch midpoint in pixels, a pinch
flag with hysteresis, and the 21 landmarks for drawing) or ``None`` when no
hand is visible.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "mediapipe", "hand_landmarker.task"
)


@dataclass
class HandState:
    point: np.ndarray         # (2,) pixel of the thumb-index midpoint
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
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        if not os.path.exists(_MODEL_PATH):
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            print(f"Downloading hand landmark model to {_MODEL_PATH} ...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=track_conf,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self._pinching = False

    def process(self, frame_bgr) -> "HandState | None":
        import cv2
        import mediapipe as mp

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            self._pinching = False
            return None

        lm = result.hand_landmarks[0]
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
        self._detector.close()
