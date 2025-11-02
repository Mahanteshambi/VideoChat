"""Basic usage example for VideoIndex."""

from pathlib import Path
from videoindex.processing import VideoProcessingPipeline, ProcessingConfig
from videoindex.config import load_settings


def main():
    """Basic usage example."""
    
    # Load settings
    settings = load_settings()
    
    # Create processing configuration
    config = ProcessingConfig(
        video_path="path/to/your/video.mp4",  # Replace with actual video path
        output_dir="./example_outputs",
        save_json=True,
        save_to_vector_db=True,
        vector_db_collection="example_annotations",
        compare_models=True,
        models_to_use=["FastVLM", "Qwen3"]  # Use specific models
    )
    
    # Create and run pipeline
    pipeline = VideoProcessingPipeline(config)
    
    try:
        results = pipeline.process()
        
        print("Processing completed successfully!")
        print(f"Processed with {len(results)} models")
        
        # Search for specific content
        search_results = pipeline.search_annotations("action scene")
        print(f"Found {len(search_results)} action scenes")
        
        # Get database statistics
        stats = pipeline.get_database_stats()
        print(f"Database contains {stats['total_documents']} documents")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
