"""Annotator modules for video processing."""

from .base import BaseAnnotator, AnnotationResult, ShotInfo
from .fastvlm import FastVLMAnnotator
from .quen2 import Quen2Annotator
from .qwen3 import Qwen3Annotator

__all__ = ["BaseAnnotator", "AnnotationResult", "ShotInfo", "FastVLMAnnotator", "Quen2Annotator", "Qwen3Annotator"]
