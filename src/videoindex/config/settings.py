"""Configuration settings management."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    model_id: str
    enabled: bool = True
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    enabled: bool = True
    collection_name: str = "video_annotations"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    output_directory: str = "./outputs"
    save_json: bool = True
    save_to_vector_db: bool = True
    compare_models: bool = True
    max_frames_per_shot: int = 5
    shot_detection_threshold: int = 27


@dataclass
class Settings:
    """Main settings configuration."""
    # Model configurations
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig(
            name="FastVLM",
            model_id="apple/FastVLM-0.5B",
            enabled=True
        ),
        ModelConfig(
            name="Quen2",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            enabled=True
        ),
        ModelConfig(
            name="Qwen3",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            enabled=True
        )
    ])
    
    # Vector database configuration
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Device configuration
    device: Optional[str] = None  # Auto-detect if None
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        return [model.name for model in self.models if model.enabled]
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        for model in self.models:
            if model.name == model_name:
                return model
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "models": [
                {
                    "name": model.name,
                    "model_id": model.model_id,
                    "enabled": model.enabled,
                    "max_tokens": model.max_tokens,
                    "temperature": model.temperature,
                    "top_p": model.top_p
                }
                for model in self.models
            ],
            "vector_db": {
                "enabled": self.vector_db.enabled,
                "collection_name": self.vector_db.collection_name,
                "persist_directory": self.vector_db.persist_directory,
                "embedding_model": self.vector_db.embedding_model
            },
            "processing": {
                "output_directory": self.processing.output_directory,
                "save_json": self.processing.save_json,
                "save_to_vector_db": self.processing.save_to_vector_db,
                "compare_models": self.processing.compare_models,
                "max_frames_per_shot": self.processing.max_frames_per_shot,
                "shot_detection_threshold": self.processing.shot_detection_threshold
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
            "device": self.device
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        models = [
            ModelConfig(**model_data)
            for model_data in data.get("models", [])
        ]
        
        vector_db = VectorDBConfig(**data.get("vector_db", {}))
        processing = ProcessingConfig(**data.get("processing", {}))
        
        return cls(
            models=models,
            vector_db=vector_db,
            processing=processing,
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
            device=data.get("device")
        )


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load settings from file and environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Default settings
    settings = Settings()
    
    # Override with environment variables
    if os.getenv("VIDEOINDEX_LOG_LEVEL"):
        settings.log_level = os.getenv("VIDEOINDEX_LOG_LEVEL")
    
    if os.getenv("VIDEOINDEX_DEVICE"):
        settings.device = os.getenv("VIDEOINDEX_DEVICE")
    
    if os.getenv("VIDEOINDEX_OUTPUT_DIR"):
        settings.processing.output_directory = os.getenv("VIDEOINDEX_OUTPUT_DIR")
    
    if os.getenv("VIDEOINDEX_VECTOR_DB_COLLECTION"):
        settings.vector_db.collection_name = os.getenv("VIDEOINDEX_VECTOR_DB_COLLECTION")
    
    # Load from config file if provided
    if config_file and Path(config_file).exists():
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            settings = Settings.from_dict(config_data)
    
    return settings


def save_settings(settings: Settings, config_file: str) -> None:
    """Save settings to file."""
    import json
    
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(settings.to_dict(), f, indent=2)
