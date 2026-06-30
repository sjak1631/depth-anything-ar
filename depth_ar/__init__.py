"""Depth-Anything based AR (occlusion + object placement) toolkit."""

from .renderer import CubeRenderer, RenderBuffers
from .object3d import Object3D
from .compositor import composite, normalize_depth, depth_to_color
from .physics import Physics
from .interaction import sample_close, near_cube, grab_move

__all__ = [
    "CubeRenderer",
    "RenderBuffers",
    "Object3D",
    "Physics",
    "composite",
    "normalize_depth",
    "depth_to_color",
    "sample_close",
    "near_cube",
    "grab_move",
]
