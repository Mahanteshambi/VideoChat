"""FastVLM annotator implementation."""

import json
import time
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from moviepy import VideoFileClip
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from loguru import logger

from .base import BaseAnnotator, AnnotationResult, ShotInfo


class FastVLMAnnotator(BaseAnnotator):
    """FastVLM annotator for video shot analysis using Apple's FastVLM model."""

    def __init__(self, model_id: str = "apple/FastVLM-0.5B"):
        super().__init__("FastVLM")
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.IMAGE_TOKEN_INDEX = -200
        self.max_frame_side = None

    def initialize(self) -> None:
        """Initialize the FastVLM model."""
        logger.info(f"Initializing Apple FastVLM model: {self.model_id}...")
        
        # Load tokenizer and model with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
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
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            if self.device != "cuda" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load with optimizations: {e}")
            # Fallback to basic loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
        
        logger.info(f"FastVLM model loaded successfully on {self.device}")

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
                            logger.debug(f"Frame {i+1}: original size {w}x{h}, max_side={max_side}, limit={self.max_frame_side}")
                            if max_side > self.max_frame_side:
                                scale = self.max_frame_side / float(max_side)
                                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                                logger.debug(f"Resizing frame {i+1} from {w}x{h} to {new_size[0]}x{new_size[1]} (scale={scale:.3f})")
                                pil_image = pil_image.resize(new_size, Image.BILINEAR)
                            else:
                                logger.debug(f"Frame {i+1}: no resize needed (max_side={max_side} <= limit={self.max_frame_side})")
                        frames.append(pil_image)
                        
                        logger.debug(f"Extracted frame {i+1}/{num_frames} at {frame_time:.2f}s")
                
                return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def extract_metadata_for_shot(self, video_path: str, shot_info: ShotInfo) -> AnnotationResult:
        """Extract metadata for a shot using Apple's FastVLM with multiple frames for better coverage."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        logger.info(f"Processing shot {shot_info.shot_number} with Apple FastVLM...")
        
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
                
                # Build chat message following Apple's FastVLM documentation
                messages = [
                    {"role": "user", "content": f"<image>\n{self.prompt}"}
                ]
                
                # Render to string (not tokens) so we can place <image> exactly
                rendered = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                pre, post = rendered.split("<image>", 1)
                
                # Tokenize the text *around* the image token (no extra specials!)
                pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
                
                # Splice in the IMAGE token id (-200) at the placeholder position
                img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
                attention_mask = torch.ones_like(input_ids, device=self.model.device)
                
                # Preprocess image via the model's own processor
                px = self.model.get_vision_tower().image_processor(images=frame, return_tensors="pt")["pixel_values"]
                px = px.to(self.model.device, dtype=self.model.dtype)
                
                # Generate response with optimizations
                with torch.no_grad():
                    generation_kwargs = {
                        "inputs": input_ids,
                        "attention_mask": attention_mask,
                        "images": px,
                        "max_new_tokens": 300,  # Sufficient for complete JSON response
                        "temperature": 0.1,
                        "do_sample": True,
                        "top_p": 0.9,
                        "pad_token_id": self.tokenizer.eos_token_id,
                    }
                    
                    # Add MPS-specific optimizations
                    if self.device == "mps":
                        generation_kwargs["use_cache"] = True
                        generation_kwargs["num_beams"] = 1
                    
                    generated_ids = self.model.generate(**generation_kwargs)
                
                # Decode the response
                response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract the response part (after the prompt)
                if rendered in response_text:
                    response_text = response_text.split(rendered)[-1].strip()
                
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
