"""Video processing pipeline for annotation extraction and vector storage."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from ..annotators import BaseAnnotator, AnnotationResult
from ..vector_db import BaseVectorDB


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
    video_path: str
    output_dir: str = "./outputs"
    save_json: bool = True
    save_to_vector_db: bool = True
    vector_db_collection: str = "video_annotations"
    compare_models: bool = True
    models_to_use: List[str] = None  # None means use all available models
    max_shots: Optional[int] = None  # None means process all shots
    max_frame_side: Optional[int] = None  # Resize frames so max(width,height) == this value
    
    def __post_init__(self):
        if self.models_to_use is None:
            self.models_to_use = ["FastVLM", "Quen2", "Qwen3"]


class VideoProcessingPipeline:
    """Main pipeline for indexing videos with multiple annotators and storing in vector database."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.console = Console()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize annotators
        self.annotators: Dict[str, BaseAnnotator] = {}
        self.vector_db: Optional[BaseVectorDB] = None
        
        self._setup_annotators()
        if config.save_to_vector_db:
            self._setup_vector_db()
    
    def _setup_annotators(self) -> None:
        """Initialize the annotator classes (not the models yet)."""
        from ..annotators import FastVLMAnnotator, Quen2Annotator, Qwen3Annotator
        
        # Store annotator classes, not instances
        self.annotator_classes = {}
        
        if "FastVLM" in self.config.models_to_use:
            self.annotator_classes["FastVLM"] = FastVLMAnnotator
        
        if "Quen2" in self.config.models_to_use:
            self.annotator_classes["Quen2"] = Quen2Annotator
        
        if "Qwen3" in self.config.models_to_use:
            self.annotator_classes["Qwen3"] = Qwen3Annotator
        
        logger.info(f"Prepared annotator classes: {list(self.annotator_classes.keys())}")
    
    def _setup_vector_db(self) -> None:
        """Initialize the vector database."""
        from ..vector_db import ChromaVectorDB
        
        self.vector_db = ChromaVectorDB(
            collection_name=self.config.vector_db_collection,
            persist_directory=str(self.output_dir / "vector_db")
        )
        self.vector_db.initialize()
        logger.info(f"Initialized vector database: {self.config.vector_db_collection}")
    
    def _process_with_annotator_sequential(self, annotator_class, model_name: str, video_path: str) -> List[AnnotationResult]:
        """Process video with a specific annotator class (load, process, unload)."""
        self.console.print(f"[bold blue]Processing with {model_name}...[/bold blue]")
        
        # Create annotator instance
        annotator = annotator_class()
        # Pass down optional frame resize hint
        try:
            setattr(annotator, "max_frame_side", self.config.max_frame_side)
        except Exception:
            pass
        
        try:
            # Initialize the model
            self.console.print(f"[yellow]Loading {model_name} model...[/yellow]")
            annotator.initialize()
            self.console.print(f"[green]✓[/green] {model_name} loaded successfully")
            
            # Process the video
            start_time = time.time()
            results = annotator.process_video(video_path, max_shots=self.config.max_shots)
            processing_time = time.time() - start_time
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            self.console.print(f"[green]✓[/green] {model_name} completed: {len(successful_results)}/{len(results)} shots successful")
            if failed_results:
                self.console.print(f"[yellow]⚠[/yellow] {len(failed_results)} shots failed")
            
            return results
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to process with {model_name}: {e}")
            logger.error(f"Processing error with {model_name}: {e}")
            return []
        
        finally:
            # Clean up the model from memory
            try:
                self.console.print(f"[yellow]Unloading {model_name} model...[/yellow]")
                if hasattr(annotator, 'model') and annotator.model is not None:
                    del annotator.model
                if hasattr(annotator, 'tokenizer') and annotator.tokenizer is not None:
                    del annotator.tokenizer
                del annotator
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                self.console.print(f"[green]✓[/green] {model_name} unloaded from memory")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup for {model_name}: {cleanup_error}")
    
    
    def _save_results(self, all_results: Dict[str, List[AnnotationResult]], video_name: str) -> None:
        """Save results to JSON files."""
        if not self.config.save_json:
            return
        
        # Save individual model results
        for model_name, results in all_results.items():
            output_file = self.output_dir / f"{video_name}_{model_name.lower()}_results.json"
            
            # Convert results to serializable format
            serializable_results = [result.to_dict() for result in results]
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved {model_name} results to: {output_file}")
        
        # Save combined results if comparing models
        if self.config.compare_models and len(all_results) > 1:
            combined_file = self.output_dir / f"{video_name}_combined_results.json"
            
            combined_data = {
                "video_name": video_name,
                "processing_timestamp": time.time(),
                "models_used": list(all_results.keys()),
                "results_by_model": {
                    model_name: [result.to_dict() for result in results]
                    for model_name, results in all_results.items()
                }
            }
            
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
            
            logger.info(f"Saved combined results to: {combined_file}")
    
    def _save_to_vector_db(self, all_results: Dict[str, List[AnnotationResult]], video_name: str) -> None:
        """Save results to vector database."""
        if not self.config.save_to_vector_db or self.vector_db is None:
            return
        
        self.console.print("[bold blue]Saving to vector database...[/bold blue]")
        
        for model_name, results in all_results.items():
            # Convert results to the format expected by vector DB
            serializable_results = [result.to_dict() for result in results]
            
            try:
                self.vector_db.add_annotation_results(serializable_results, f"{video_name}_{model_name}")
                self.console.print(f"[green]✓[/green] Saved {model_name} results to vector database")
            except Exception as e:
                self.console.print(f"[red]✗[/red] Failed to save {model_name} results to vector database: {e}")
                logger.error(f"Vector DB save error for {model_name}: {e}")
    
    def _display_summary(self, all_results: Dict[str, List[AnnotationResult]], video_name: str) -> None:
        """Display processing summary."""
        self.console.print("\n[bold green]Processing Summary[/bold green]")
        self.console.print("=" * 50)
        
        # Create summary table
        table = Table(title=f"Results for {video_name}")
        table.add_column("Model", style="cyan")
        table.add_column("Total Shots", justify="right")
        table.add_column("Successful", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Avg Time/Shot", justify="right")
        
        for model_name, results in all_results.items():
            total_shots = len(results)
            successful = len([r for r in results if r.success])
            failed = total_shots - successful
            
            if total_shots > 0:
                avg_time = sum(r.processing_time_seconds for r in results) / total_shots
                avg_time_str = f"{avg_time:.2f}s"
            else:
                avg_time_str = "N/A"
            
            table.add_row(
                model_name,
                str(total_shots),
                str(successful),
                str(failed),
                avg_time_str
            )
        
        self.console.print(table)
        
        # Display vector database info
        if self.vector_db:
            try:
                stats = self.vector_db.get_collection_stats()
                self.console.print(f"\n[bold blue]Vector Database Stats[/bold blue]")
                self.console.print(f"Total documents: {stats['total_documents']}")
                self.console.print(f"Collection: {stats['collection_name']}")
            except Exception as e:
                logger.error(f"Error getting vector DB stats: {e}")
    
    def process(self) -> Dict[str, List[AnnotationResult]]:
        """Main video indexing pipeline."""
        video_path = Path(self.config.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_name = video_path.stem
        
        self.console.print(f"[bold green]Starting video indexing pipeline[/bold green]")
        self.console.print(f"Video: {video_path}")
        self.console.print(f"Models: {', '.join(self.annotator_classes.keys())}")
        self.console.print(f"Output directory: {self.output_dir}")
        self.console.print(f"[yellow]Processing models sequentially to save memory[/yellow]")
        
        # Process with each annotator sequentially (load, process, unload)
        all_results = {}
        
        for model_name, annotator_class in self.annotator_classes.items():
            try:
                results = self._process_with_annotator_sequential(annotator_class, model_name, str(video_path))
                if results:  # Only add if we got results
                    all_results[model_name] = results
            except Exception as e:
                self.console.print(f"[red]✗[/red] Failed to process with {model_name}: {e}")
                logger.error(f"Processing error with {model_name}: {e}")
        
        if not all_results:
            raise RuntimeError("No models could process the video successfully")
        
        # Save results
        self._save_results(all_results, video_name)
        self._save_to_vector_db(all_results, video_name)
        
        # Display summary
        self._display_summary(all_results, video_name)
        
        return all_results
    
    def search_annotations(self, query: str, model_filter: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search through indexed video annotations in the vector database."""
        if self.vector_db is None:
            raise RuntimeError("Vector database not initialized")
        
        return self.vector_db.search(query, top_k=top_k, model_filter=model_filter)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        if self.vector_db is None:
            raise RuntimeError("Vector database not initialized")
        
        return self.vector_db.get_collection_stats()
