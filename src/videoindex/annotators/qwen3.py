"""Qwen3-VL annotator implementation for video understanding."""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from moviepy import VideoFileClip
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from loguru import logger

from .base import BaseAnnotator, AnnotationResult, ShotInfo


class Qwen3Annotator(BaseAnnotator):
    """Qwen3-VL annotator for video shot analysis using Qwen3-VL model."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct"):
        super().__init__("Qwen3")
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.max_frame_side = None

    def initialize(self) -> None:
        """Initialize the Qwen3-VL model."""
        logger.info(f"Initializing Qwen3-VL model: {self.model_id}...")
        
        try:
            # Load processor and model for Qwen3-VL
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # Optimize model loading
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Set appropriate dtype based on device
            if self.device == "cuda":
                model_kwargs["dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            elif self.device == "mps":
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float32
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, 
                **model_kwargs
            )
            
            if self.device != "cuda" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Qwen3-VL model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL model: {e}")
            raise

    def _extract_frames_from_shot(self, video_path: str, shot_info: ShotInfo, num_frames: int = 3) -> List[Image.Image]:
        """Extract multiple representative frames from a shot for better coverage."""
        frames = []
        try:
            with VideoFileClip(video_path) as video:
                start_time = shot_info.start_time_seconds
                end_time = shot_info.end_time_seconds
                duration = end_time - start_time
                
                # For very long shots (>10s), extract more frames
                if duration > 10:
                    num_frames = 5
                elif duration > 5:
                    num_frames = 4
                
                # Extract frames at different positions
                for i in range(num_frames):
                    if num_frames == 1:
                        frame_time = start_time + duration / 2  # Middle frame
                    else:
                        frame_time = start_time + (duration * i / (num_frames - 1))
                    
                    if frame_time < video.duration:
                        frame = video.get_frame(frame_time)
                        pil_image = Image.fromarray(frame).convert("RGB")
                        
                        # Optional downscale to speed up inference
                        if self.max_frame_side and self.max_frame_side > 0:
                            w, h = pil_image.size
                            max_side = max(w, h)
                            if max_side > self.max_frame_side:
                                scale = self.max_frame_side / float(max_side)
                                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                                pil_image = pil_image.resize(new_size, Image.BILINEAR)
                                logger.debug(f"Resized frame {i+1} from {w}x{h} to {new_size[0]}x{new_size[1]}")
                        
                        frames.append(pil_image)
                        logger.debug(f"Extracted frame {i+1}/{num_frames} at {frame_time:.2f}s")
                
                return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def extract_metadata_for_shot(self, video_path: str, shot_info: ShotInfo) -> AnnotationResult:
        """Extract metadata for a shot using Qwen3-VL with proper frame analysis."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        logger.info(f"Processing shot {shot_info.shot_number} with Qwen3-VL...")
        
        # Extract multiple frames from the shot
        frames = self._extract_frames_from_shot(video_path, shot_info)
        if not frames:
            return AnnotationResult(
                shot_info=shot_info,
                metadata={"error": "Failed to extract frames from shot"},
                model_name=self.model_name,
                processing_time_seconds=0.0,
                success=False,
                error_message="Failed to extract frames from shot"
            )

        # Process frames - use the first frame as primary, analyze others as additional context
        all_responses = []
        
        # Clear any cached states to avoid numerical accumulation issues
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # For Qwen3-VL, we can process multiple images in a single call for better understanding
            # But start with single frame processing for reliability
            primary_frame = frames[0]
            
            logger.info(f"Processing primary frame for shot {shot_info.shot_number}")
            
            # Build messages following Qwen3-VL format
            # Use chat template first, then process with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": primary_frame,
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                }
            ]
            
            # Apply chat template to format the messages
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process with both text and image
            inputs = self.processor(
                text=text,
                images=[primary_frame],
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response with more stable parameters
            with torch.no_grad():
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,  # More tokens for detailed descriptions
                        temperature=0.3,  # Lower temperature for more stable outputs (reduced from 0.7)
                        do_sample=True,
                        top_p=0.85,  # Slightly lower top_p for stability
                        pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                    )
                except RuntimeError as e:
                    if 'inf' in str(e) or 'nan' in str(e) or 'probability tensor' in str(e):
                        # Try with even more conservative parameters
                        logger.warning(f"Numerical instability detected, retrying with conservative parameters: {e}")
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.1,  # Very low temperature
                            do_sample=False,  # Use greedy decoding for stability
                            pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                        )
                    else:
                        raise
            
            # Decode response - remove input tokens
            input_length = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[:, input_length:]
            response_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            logger.debug(f"Raw response length: {len(response_text)}")
            logger.debug(f"Response preview: {response_text[:200]}...")
            
            # Parse JSON from response
            try:
                cleaned_text = response_text.strip()
                
                # Remove markdown code blocks if present
                if cleaned_text.startswith('```'):
                    lines = cleaned_text.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned_text = '\n'.join(lines)
                
                # Extract JSON object
                json_start = cleaned_text.find('{')
                json_end = cleaned_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = cleaned_text[json_start:json_end]
                    frame_response = json.loads(json_str)
                    
                    # Verify we got actual content (not placeholder)
                    description = frame_response.get("ShotDescription", "")
                    if description and len(description) > 20:  # Must be substantial
                        frame_response["frame_number"] = 1
                        frame_response["frame_time"] = shot_info.start_time_seconds
                        all_responses.append(frame_response)
                        logger.info(f"Successfully parsed JSON from Qwen3-VL response")
                    else:
                        logger.warning(f"Response seems too short or placeholder-like: {description}")
                        all_responses.append({
                            "ShotDescription": response_text[:500],
                            "raw_response": response_text,
                            "frame_number": 1,
                            "parse_warning": "Response may not be properly formatted"
                        })
                else:
                    # No JSON found, store as raw text
                    logger.warning("No JSON object found in response, storing as raw text")
                    all_responses.append({
                        "ShotDescription": response_text[:500],
                        "raw_response": response_text,
                        "frame_number": 1,
                        "parse_error": "Could not extract JSON"
                    })
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.debug(f"Problematic text: {response_text[:500]}")
                all_responses.append({
                    "ShotDescription": response_text[:500],
                    "raw_response": response_text,
                    "frame_number": 1,
                    "parse_error": str(e)
                })
        
        except Exception as e:
            logger.error(f"Error processing shot {shot_info.shot_number}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            all_responses.append({
                "error": f"Processing error: {str(e)}",
                "frame_number": 1
            })
        
        # Process additional frames if available
        if len(frames) > 1 and all_responses:
            logger.info(f"Processing {len(frames) - 1} additional frames for context")
            for i, frame in enumerate(frames[1:], start=2):
                try:
                    # Quick analysis of additional frames
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": frame,
                                },
                                {
                                    "type": "text",
                                    "text": "Briefly describe what you see in this frame in one sentence.",
                                },
                            ],
                        }
                    ]
                    
                    inputs = self.processor(
                        text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                        images=[frame],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                        )
                    
                    input_length = inputs['input_ids'].shape[1]
                    generated_ids = generated_ids[:, input_length:]
                    response_text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )[0].strip()
                    
                    if response_text and len(response_text) > 10:
                        frame_time = shot_info.start_time_seconds + (
                            (shot_info.end_time_seconds - shot_info.start_time_seconds) * (i - 1) / (len(frames) - 1)
                        )
                        all_responses.append({
                            "ShotDescription": response_text,
                            "frame_number": i,
                            "frame_time": frame_time
                        })
                        
                except Exception as e:
                    logger.debug(f"Error processing additional frame {i}: {e}")
                    continue
        
        # Return results
        if all_responses:
            primary_response = all_responses[0]
            if len(all_responses) > 1:
                primary_response["additional_frames"] = all_responses[1:]
            
            logger.info(f"Successfully processed shot {shot_info.shot_number} with {len(frames)} frames")
            return AnnotationResult(
                shot_info=shot_info,
                metadata=primary_response,
                model_name=self.model_name,
                processing_time_seconds=0.0,  # Will be set by caller
                success=True
            )
        else:
            logger.error(f"No output generated for shot {shot_info.shot_number}")
            return AnnotationResult(
                shot_info=shot_info,
                metadata={"error": "No output generated"},
                model_name=self.model_name,
                processing_time_seconds=0.0,
                success=False,
                error_message="No output generated"
            )

    def detect_shots(self, video_path: str) -> List[ShotInfo]:
        """Detect shots in video using PySceneDetect."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27))
        scene_manager.detect_scenes(video, show_progress=True)
        shot_list = scene_manager.get_scene_list()
        
        return [
            ShotInfo(
                shot_number=i + 1,
                start_time_seconds=shot[0].get_seconds(),
                end_time_seconds=shot[1].get_seconds(),
                duration_seconds=shot[1].get_seconds() - shot[0].get_seconds()
            )
            for i, shot in enumerate(shot_list)
        ]

