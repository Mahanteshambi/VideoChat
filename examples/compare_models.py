"""Example comparing FastVLM vs Qwen3 model performance."""

import json
import time
from pathlib import Path
from videoindex.processing import VideoProcessingPipeline, ProcessingConfig
from videoindex.config import load_settings


def compare_models(video_path: str, output_dir: str = "./comparison_outputs", models: list = None):
    """Compare vision-language models on the same video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory for output files
        models: List of model names to compare (default: ["FastVLM", "Qwen3"])
    """
    if models is None:
        models = ["FastVLM", "Qwen3"]
    
    # Load settings
    settings = load_settings()
    
    results = {}
    
    for model_name in models:
        print(f"\nProcessing with {model_name}...")
        config = ProcessingConfig(
            video_path=video_path,
            output_dir=output_dir,
            save_json=True,
            save_to_vector_db=False,  # Skip vector DB for individual runs
            compare_models=False,
            models_to_use=[model_name]
        )
        
        pipeline = VideoProcessingPipeline(config)
        start_time = time.time()
        model_results = pipeline.process()
        model_time = time.time() - start_time
        
        results[model_name] = {
            "results": model_results,
            "total_time": model_time,
            "shots_processed": len(model_results.get(model_name, [])),
            "successful_shots": len([r for r in model_results.get(model_name, []) if r.success])
        }
    
    # Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"  Total processing time: {model_results['total_time']:.2f} seconds")
        print(f"  Shots processed: {model_results['shots_processed']}")
        print(f"  Successful shots: {model_results['successful_shots']}")
        
        if model_results['shots_processed'] > 0:
            success_rate = model_results['successful_shots'] / model_results['shots_processed'] * 100
            avg_time_per_shot = model_results['total_time'] / model_results['shots_processed']
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average time per shot: {avg_time_per_shot:.2f} seconds")
    
    # Save comparison results
    comparison_file = Path(output_dir) / "model_comparison.json"
    comparison_data = {
        "video_path": video_path,
        "comparison_timestamp": time.time(),
        "results": {
            model_name: {
                "total_time": model_results["total_time"],
                "shots_processed": model_results["shots_processed"],
                "successful_shots": model_results["successful_shots"],
                "success_rate": model_results["successful_shots"] / model_results["shots_processed"] * 100 if model_results["shots_processed"] > 0 else 0,
                "avg_time_per_shot": model_results["total_time"] / model_results["shots_processed"] if model_results["shots_processed"] > 0 else 0
            }
            for model_name, model_results in results.items()
        }
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_file}")
    
    return results


def main():
    """Main function for model comparison example."""
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "path/to/your/video.mp4"  # Replace with actual video path
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("Usage: python compare_models.py <video_path>")
        print("Or update the video_path variable in the script.")
        return
    
    models = ["FastVLM", "Qwen3"]
    if len(sys.argv) > 2:
        models = [m.strip() for m in sys.argv[2].split(",")]
    
    try:
        results = compare_models(video_path, models=models)
        print("\nModel comparison completed successfully!")
        
    except Exception as e:
        print(f"Error during model comparison: {e}")


if __name__ == "__main__":
    main()
