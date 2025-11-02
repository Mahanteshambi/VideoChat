"""Base annotator interface and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ShotInfo:
    """Information about a video shot."""
    shot_number: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    
    def __post_init__(self):
        if self.duration_seconds <= 0:
            self.duration_seconds = self.end_time_seconds - self.start_time_seconds


@dataclass
class AnnotationResult:
    """Result of video annotation."""
    shot_info: ShotInfo
    metadata: Dict[str, Any]
    model_name: str
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "shot_info": {
                "shot_number": self.shot_info.shot_number,
                "start_time_seconds": self.shot_info.start_time_seconds,
                "end_time_seconds": self.shot_info.end_time_seconds,
                "duration_seconds": self.shot_info.duration_seconds,
            },
            "metadata": self.metadata,
            "model_name": self.model_name,
            "processing_time_seconds": self.processing_time_seconds,
            "success": self.success,
            "error_message": self.error_message,
        }


class BaseAnnotator(ABC):
    """Base class for video annotators."""
    
    # Model-specific prompts - optimized for each architecture but equal length
    # Extremely explicit to prevent common issues (embedding JSON in descriptions, markdown wrapping)
    
    FASTVLM_PROMPT = """Analyze this image and return a JSON object. Rules:
1. Write a brief sentence describing the scene for "ShotDescription"
2. Fill ALL other fields with relevant data from the image
3. Return ONLY raw JSON (no markdown, no code blocks, no extra text)

{
"ShotDescription": "A one-sentence description of what you see",
"GenreCues": [{"genre_hint": "Animation", "prominence_in_shot": "Cartoon style"}],
"SubgenreCues": ["Family"],
"AdjectiveTheme": ["Heartwarming"],
"Mood": ["Joyful"],
"SettingContext": ["Home"],
"ContentDescriptors": ["Characters"],
"LocationHints_Regional": ["Urban"],
"LocationHints_International": ["USA"],
"SearchKeywords": ["family", "animation"]
}"""
    
    QWEN2_PROMPT = """Analyze this image and return a JSON object. Rules:
1. Write a brief sentence describing the scene for "ShotDescription"
2. Fill ALL other fields with relevant data from the image
3. Return ONLY raw JSON (no markdown, no code blocks, no extra text)

{
"ShotDescription": "A one-sentence description of what you see",
"GenreCues": [{"genre_hint": "Animation", "prominence_in_shot": "Cartoon style"}],
"SubgenreCues": ["Family"],
"AdjectiveTheme": ["Heartwarming"],
"Mood": ["Joyful"],
"SettingContext": ["Home"],
"ContentDescriptors": ["Characters"],
"LocationHints_Regional": ["Urban"],
"LocationHints_International": ["USA"],
"SearchKeywords": ["family", "animation"]
}"""
    
    QWEN3_PROMPT = """Carefully observe this image. You must analyze the specific visual content you see - the actual objects, people, actions, colors, and scene composition. Describe EXACTLY what is visible in this particular image, not a generic description.

After analyzing, return a JSON object with:
1. "ShotDescription": A specific one-sentence description of what you actually see in this image (e.g., "A young girl with pink hair sits on a bed holding a tablet", NOT a generic phrase)
2. Fill ALL other fields based on what you observe in this specific image

Return ONLY raw JSON (no markdown, no code blocks, no extra text before or after):

{
"ShotDescription": "describe exactly what you see in this specific image",
"GenreCues": [{"genre_hint": "infer from image", "prominence_in_shot": "describe visual style"}],
"SubgenreCues": ["infer from content"],
"AdjectiveTheme": ["infer from mood"],
"Mood": ["infer from scene"],
"SettingContext": ["infer from background/environment"],
"ContentDescriptors": ["list visible elements"],
"LocationHints_Regional": ["infer from visual cues"],
"LocationHints_International": ["infer from visual cues"],
"SearchKeywords": ["extract relevant keywords from visible content"]
}"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = self._get_device()
        # Set model-specific prompt
        self.prompt = self._get_model_prompt()
    
    def _get_model_prompt(self) -> str:
        """Get the appropriate prompt for this model."""
        if "FastVLM" in self.model_name or "fastvlm" in self.model_name.lower():
            return self.FASTVLM_PROMPT
        elif "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
            return self.QWEN3_PROMPT
        elif "Quen2" in self.model_name or "qwen2" in self.model_name.lower() or "qwen" in self.model_name.lower():
            return self.QWEN2_PROMPT
        else:
            # Fallback to FastVLM prompt for unknown models
            return self.FASTVLM_PROMPT
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the annotator model."""
        pass
    
    @abstractmethod
    def extract_metadata_for_shot(self, video_path: str, shot_info: ShotInfo) -> AnnotationResult:
        """Extract metadata for a single shot."""
        pass
    
    @abstractmethod
    def detect_shots(self, video_path: str) -> List[ShotInfo]:
        """Detect shots in a video."""
        pass
    
    def process_video(self, video_path: str, max_shots: Optional[int] = None) -> List[AnnotationResult]:
        """Process all shots in a video."""
        import time
        from loguru import logger
        
        logger.info(f"Starting {self.model_name} processing for: {video_path}")
        
        # Detect shots
        shots = self.detect_shots(video_path)
        logger.info(f"Detected {len(shots)} shots in video")
        
        # Limit shots if max_shots is specified
        if max_shots is not None and max_shots > 0:
            shots = shots[:max_shots]
            logger.info(f"Limited to processing {len(shots)} shots (max_shots={max_shots})")
        
        results = []
        for shot in shots:
            logger.info(f"Processing shot {shot.shot_number}/{len(shots)}")
            
            start_time = time.time()
            try:
                result = self.extract_metadata_for_shot(video_path, shot)
                result.processing_time_seconds = time.time() - start_time
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing shot {shot.shot_number}: {e}")
                error_result = AnnotationResult(
                    shot_info=shot,
                    metadata={},
                    model_name=self.model_name,
                    processing_time_seconds=time.time() - start_time,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        logger.info(f"Completed {self.model_name} processing. {len([r for r in results if r.success])}/{len(results)} shots successful")
        return results
