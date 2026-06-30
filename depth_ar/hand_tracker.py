"""MediaPipe hand tracking + whole-hand grasp detection (Tasks API, >=0.10).

Isolated here so the rest of the package imports without MediaPipe installed.
``process`` returns a list of :class:`HandState` (one per detected hand) with the
palm-centre point in pixels, a *grasp* flag (the whole hand closing into a fist,
with hysteresis), a Left/Right label for stable identity, and the 21 landmarks.
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
    point: np.ndarray         # (2,) palm-centre pixel (grab anchor)
    grasping: bool
    landmarks_px: np.ndarray  # (21, 2) pixel landmark coordinates
    label: str = "?"          # "Left" / "Right" (stable identity key)


class HandTracker:
    WRIST = 0
    MCPS = (5, 9, 13, 17)              # finger knuckles (palm)
    PALM = (0, 5, 9, 13, 17)          # wrist + knuckles -> palm centroid
    FINGERTIPS = (8, 12, 16, 20)      # index..pinky tips (not thumb)
    PALM_REF = 9                       # middle-finger MCP

    def __init__(
        self,
        max_hands: int = 2,
        det_conf: float = 0.6,
        track_conf: float = 0.5,
        grasp_on: float = 1.0,
        grasp_off: float = 1.3,
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
        self.grasp_on = grasp_on
        self.grasp_off = grasp_off
        self._grasp_state: dict[str, bool] = {}   # per-label hysteresis

    def process(self, frame_bgr) -> "list[HandState]":
        import cv2
        import mediapipe as mp

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            self._grasp_state.clear()
            return []

        hands: list[HandState] = []
        seen: dict[str, bool] = {}
        for i, lm in enumerate(result.hand_landmarks):
            pts = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)
            label = "?"
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
            # Disambiguate if both hands share a label this frame.
            if label in seen:
                label = f"{label}{i}"
            seen[label] = True

            # Whole-hand grasp: fingertips fold toward the palm centre.
            size = np.linalg.norm(pts[self.WRIST] - pts[self.PALM_REF]) + 1e-6
            palm = pts[self.PALM_REF]
            tip_dist = np.mean([np.linalg.norm(pts[t] - palm) for t in self.FINGERTIPS])
            ratio = float(tip_dist / size)

            prev = self._grasp_state.get(label, False)
            grasping = ratio < self.grasp_off if prev else ratio < self.grasp_on
            self._grasp_state[label] = grasping

            point = pts[list(self.PALM)].mean(axis=0)
            hands.append(HandState(point=point, grasping=grasping,
                                   landmarks_px=pts, label=label))

        # Forget stale labels so hysteresis doesn't leak between appearances.
        self._grasp_state = {h.label: self._grasp_state[h.label] for h in hands}
        return hands

    def close(self) -> None:
        self._detector.close()
