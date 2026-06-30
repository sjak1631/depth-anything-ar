"""Depth-Anything based AR (occlusion + object placement) toolkit."""

from .renderer import SphereRenderer, RenderBuffers
from .object3d import Object3D
from .compositor import composite, normalize_depth, metric_closeness, depth_to_color
from .physics import Physics
from .interaction import (
    sample_close, sample_close_nearest, near_cube, grab_move, VelocityTracker,
)

__all__ = [
    "SphereRenderer",
    "RenderBuffers",
    "Object3D",
    "Physics",
    "composite",
    "normalize_depth",
    "metric_closeness",
    "depth_to_color",
    "sample_close",
    "sample_close_nearest",
    "near_cube",
    "grab_move",
    "VelocityTracker",
]
