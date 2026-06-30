"""MediaPipe pose tracking for foot/kick interaction (Tasks API, >=0.10).

Isolated so the package imports without MediaPipe installed. ``process`` returns
a list of :class:`FootState` (one per visible foot) with the toe point in pixels,
a Left/Right label, and a few landmarks on the foot for robust depth sampling.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "mediapipe", "pose_landmarker.task"
)


@dataclass
class FootState:
    point: np.ndarray         # (2,) toe (foot-index) pixel — the kicking point
    landmarks_px: np.ndarray  # (3, 2) toe / ankle / heel pixels (depth probes)
    label: str = "?"          # "L" / "R"


class PoseTracker:
    # (label, foot_index, ankle, heel) landmark ids.
    _FEET = (("L", 31, 27, 29), ("R", 32, 28, 30))

    def __init__(self, det_conf: float = 0.5, track_conf: float = 0.5,
                 min_visibility: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        if not os.path.exists(_MODEL_PATH):
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            print(f"Downloading pose landmark model to {_MODEL_PATH} ...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self._detector = PoseLandmarker.create_from_options(options)
        self.min_visibility = min_visibility

    def process(self, frame_bgr) -> "list[FootState]":
        import cv2
        import mediapipe as mp

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.pose_landmarks:
            return []

        lm = result.pose_landmarks[0]
        feet: list[FootState] = []
        for label, toe, ankle, heel in self._FEET:
            if min(lm[toe].visibility, lm[ankle].visibility) < self.min_visibility:
                continue
            pts = np.array([[lm[i].x * W, lm[i].y * H] for i in (toe, ankle, heel)],
                           dtype=np.float32)
            feet.append(FootState(point=pts[0], landmarks_px=pts, label=label))
        return feet

    def close(self) -> None:
        self._detector.close()
