"""Depth-Anything V2 wrapper using Hugging Face transformers.

Loads ``depth-anything/Depth-Anything-V2-Small-hf`` and returns a per-frame
relative depth map (higher value = nearer). Inference can run at a reduced
resolution for speed and is upsampled back to the frame size.
"""

from __future__ import annotations

import numpy as np


class DepthEstimator:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
        fp16: bool = True,
        infer_size: int = 392,
    ):
        # Imported lazily so the numpy-only modules stay torch-free.
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16) and device == "cuda"
        self.infer_size = int(infer_size)

        def _load(cls, **kw):
            try:
                return cls.from_pretrained(model_name, local_files_only=True, **kw)
            except Exception:
                return cls.from_pretrained(model_name, **kw)

        self.processor = _load(AutoImageProcessor)
        model = _load(AutoModelForDepthEstimation)
        model = model.to(device).eval()
        if self.fp16:
            model = model.half()
        self.model = model

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 depth map (H, W) for a BGR frame. Higher = nearer."""
        import cv2

        torch = self.torch
        H, W = frame_bgr.shape[:2]

        # Optional downscale for speed; keep aspect ratio.
        scale = self.infer_size / max(H, W) if self.infer_size else 1.0
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (round(W * scale), round(H * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.fp16:
            pixel_values = pixel_values.half()

        with torch.inference_mode():
            predicted = self.model(pixel_values=pixel_values).predicted_depth

        # predicted: (1, h, w) -> upsample to full frame size.
        depth = torch.nn.functional.interpolate(
            predicted.unsqueeze(1).float(),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        return depth.cpu().numpy().astype(np.float32)
