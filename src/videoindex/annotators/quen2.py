"""Quen2 annotator implementation."""

import json
import time
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from moviepy import VideoFileClip
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from loguru import logger

from .base import BaseAnnotator, AnnotationResult, ShotInfo


class Quen2Annotator(BaseAnnotator):
    """Quen2 annotator for video shot analysis using Qwen2-VL model."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__("Quen2")
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.max_frame_side = None

    def initialize(self) -> None:
        """Initialize the Quen2 model."""
        logger.info(f"Initializing Quen2 model: {self.model_id}...")
        
        # Load tokenizer, processor and model with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Load processor for Qwen2-VL models
        if "qwen2-vl" in self.model_id.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Optimize model loading
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Set appropriate dtype based on device
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        try:
            # Try Qwen2VLForConditionalGeneration first for Qwen2-VL models
            if "qwen2-vl" in self.model_id.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            
            if self.device != "cuda" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load with optimizations: {e}")
            # Fallback to basic loading
            if "qwen2-vl" in self.model_id.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to(self.device)
        
        logger.info(f"Quen2 model loaded successfully on {self.device}")

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
                        pil_image = Image.fromarray(frame)
                        # Optional downscale to speed up inference
                        if self.max_frame_side and self.max_frame_side > 0:
                            w, h = pil_image.size
                            max_side = max(w, h)
                            if max_side > self.max_frame_side:
                                scale = self.max_frame_side / float(max_side)
                                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                                pil_image = pil_image.resize(new_size, Image.BILINEAR)
                        frames.append(pil_image)
                        
                        logger.debug(f"Extracted frame {i+1}/{num_frames} at {frame_time:.2f}s")
                
                return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def extract_metadata_for_shot(self, video_path: str, shot_info: ShotInfo) -> AnnotationResult:
        """Extract metadata for a shot using Quen2 with multiple frames for better coverage."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        logger.info(f"Processing shot {shot_info.shot_number} with Quen2...")
        
        # Extract multiple frames from the shot for better coverage
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

        # Process each frame and combine results
        all_responses = []
        for i, frame in enumerate(frames):
            try:
                logger.info(f"Processing frame {i+1}/{len(frames)} for shot {shot_info.shot_number}")
                
                # Build chat message for Quen2
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame},
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ]
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Process image and text for Qwen2-VL
                if "qwen2-vl" in self.model_id.lower() and self.processor is not None:
                    # Use Qwen2VL specific processing with proper image handling
                    try:
                        # Use the processor to handle both text and images
                        inputs = self.processor(
                            text=text,
                            images=[frame],
                            return_tensors="pt"
                        )
                        
                        # Debug: print the structure of inputs
                        logger.debug(f"Processor inputs keys: {list(inputs.keys())}")
                        for key, value in inputs.items():
                            if hasattr(value, 'shape'):
                                logger.debug(f"  {key}: shape {value.shape}, type {type(value)}")
                            else:
                                logger.debug(f"  {key}: type {type(value)}")
                        
                        # Move to device
                        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                        
                        # Generate response
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=300,  # Sufficient for complete JSON response
                                temperature=0.1,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                        
                        # Decode the response - handle different input structures
                        # Try to find input_ids in the inputs
                        input_ids = None
                        if 'input_ids' in inputs:
                            input_ids = inputs['input_ids']
                        else:
                            # Look for any tensor that could be input_ids
                            for key, value in inputs.items():
                                if hasattr(value, 'shape') and len(value.shape) == 2 and value.shape[0] == 1:
                                    input_ids = value
                                    logger.debug(f"Using {key} as input_ids with shape {value.shape}")
                                    break
                        
                        if input_ids is not None:
                            # Remove input tokens from generated tokens
                            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
                            ]
                            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        else:
                            # If we can't find input_ids, just decode the full output
                            logger.warning("Could not find input_ids, decoding full output")
                            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            
                    except Exception as vision_error:
                        logger.warning(f"Vision processing failed, trying alternative approach: {vision_error}")
                        # Fallback: try without vision processing
                        inputs = self.tokenizer(
                            text=text,
                            return_tensors="pt"
                        )
                        
                        # Move to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Generate response
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=300,  # Sufficient for complete JSON response
                                temperature=0.1,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                        
                        # Decode the response - save input_length first to avoid variable shadowing
                        input_length = inputs['input_ids'].shape[1]
                        generated_ids = [
                            output_ids[input_length:] for output_ids in generated_ids
                        ]
                        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    # Fallback for other models
                    inputs = self.tokenizer(
                        text=text,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate response
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=300,  # Increased to match other paths
                            temperature=0.1,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    # Decode the response - save input_length first to avoid variable shadowing
                    input_length = inputs['input_ids'].shape[1]
                    generated_ids = [
                        output_ids[input_length:] for output_ids in generated_ids
                    ]
                    response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if response_text:
                    # Try to parse JSON from the response
                    try:
                        # Strip markdown code blocks if present
                        cleaned_text = response_text.strip()
                        if cleaned_text.startswith('```'):
                            # Remove opening ```json or ``` and closing ```
                            lines = cleaned_text.split('\n')
                            if lines[0].startswith('```'):
                                lines = lines[1:]  # Remove first line
                            if lines and lines[-1].strip() == '```':
                                lines = lines[:-1]  # Remove last line
                            cleaned_text = '\n'.join(lines)
                        
                        # Extract JSON from the response (it might have extra text)
                        json_start = cleaned_text.find('{')
                        json_end = cleaned_text.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_str = cleaned_text[json_start:json_end]
                            frame_response = json.loads(json_str)
                            frame_response["frame_number"] = i + 1
                            frame_response["frame_time"] = shot_info.start_time_seconds + (shot_info.end_time_seconds - shot_info.start_time_seconds) * i / (len(frames) - 1)
                            all_responses.append(frame_response)
                        else:
                            # If no JSON found, return the raw text with a wrapper
                            frame_response = {"ShotDescription": response_text, "raw_response": response_text, "frame_number": i + 1}
                            all_responses.append(frame_response)
                            
                    except json.JSONDecodeError:
                        # If JSON parsing fails, return the raw text with error info
                        frame_response = {"ShotDescription": response_text, "parsing_error": True, "raw_response": response_text, "frame_number": i + 1}
                        all_responses.append(frame_response)
                
            except Exception as e:
                logger.error(f"Error processing frame {i+1} for shot {shot_info.shot_number}: {e}")
                all_responses.append({"error": f"Frame {i+1} processing error: {str(e)}", "frame_number": i + 1})
        
        # Combine responses from all frames
        if all_responses:
            # Use the first successful response as primary, add others as additional frames
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
