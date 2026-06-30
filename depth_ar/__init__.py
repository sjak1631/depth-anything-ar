"""Depth-Anything based AR (occlusion + object placement) toolkit."""

from .renderer import CubeRenderer, RenderBuffers
from .object3d import Object3D
from .compositor import composite, normalize_depth, depth_to_color

__all__ = [
    "CubeRenderer",
    "RenderBuffers",
    "Object3D",
    "composite",
    "normalize_depth",
    "depth_to_color",
]
