"""Command-line interface for VideoIndex."""

import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from .config import load_settings, save_settings, Settings
from .processing import VideoProcessingPipeline, ProcessingConfig

app = typer.Typer(
    name="videoindex",
    help="Modular video indexing and search system using FastVLM vs Quen2 with vector database storage",
    add_completion=False
)
console = Console()


@app.command()
def process(
    video_path: str = typer.Argument(..., help="Path to the video file to index"),
    output_dir: str = typer.Option("./outputs", "--output", "-o", help="Output directory for results"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated list of models to use (FastVLM,Quen2,Qwen3)"),
    no_vector_db: bool = typer.Option(False, "--no-vector-db", help="Skip saving to vector database"),
    no_json: bool = typer.Option(False, "--no-json", help="Skip saving JSON results"),
    collection_name: str = typer.Option("video_annotations", "--collection", "-c", help="Vector database collection name"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to configuration file"),
    max_shots: Optional[int] = typer.Option(None, "--max-shots", help="Maximum number of shots to process (useful for testing)"),
    max_frame_side: Optional[int] = typer.Option(None, "--max-frame-side", help="Resize frames so max(width,height)=N before inference"),
):
    """Index a video file by extracting and storing annotations for search."""
    
    # Load settings
    settings = load_settings(config_file)
    
    # Parse models
    models_to_use = None
    if models:
        models_to_use = [m.strip() for m in models.split(",")]
        # Validate models
        available_models = settings.get_enabled_models()
        invalid_models = [m for m in models_to_use if m not in available_models]
        if invalid_models:
            console.print(f"[red]Error: Invalid models: {invalid_models}[/red]")
            console.print(f"Available models: {', '.join(available_models)}")
            raise typer.Exit(1)
    
    # Create processing configuration
    processing_config = ProcessingConfig(
        video_path=video_path,
        output_dir=output_dir,
        save_json=not no_json,
        save_to_vector_db=not no_vector_db,
        vector_db_collection=collection_name,
        compare_models=True,
        models_to_use=models_to_use,
        max_shots=max_shots,
        max_frame_side=max_frame_side
    )
    
    # Display configuration
    max_shots_info = f"Max Shots: {max_shots}" if max_shots else "Max Shots: All"
    resize_info = f"Frame Resize: max_side={max_frame_side}" if max_frame_side else "Frame Resize: None"
    console.print(Panel.fit(
        f"[bold blue]VideoIndex Processing Configuration[/bold blue]\n"
        f"Video: {video_path}\n"
        f"Output: {output_dir}\n"
        f"Models: {', '.join(models_to_use or settings.get_enabled_models())}\n"
        f"{max_shots_info}\n"
        f"{resize_info}\n"
        f"Vector DB: {'Yes' if not no_vector_db else 'No'}\n"
        f"JSON Output: {'Yes' if not no_json else 'No'}",
        title="Configuration"
    ))
    
    try:
        # Create and run pipeline
        pipeline = VideoProcessingPipeline(processing_config)
        results = pipeline.process()
        
        console.print("\n[bold green]✓ Processing completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[red]✗ Processing failed: {e}[/red]")
        logger.error(f"Processing error: {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query to find relevant video content"),
    collection_name: str = typer.Option("video_annotations", "--collection", "-c", help="Vector database collection name"),
    model_filter: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results to return"),
    output_dir: str = typer.Option("./outputs", "--output", "-o", help="Output directory containing vector database"),
):
    """Search through indexed videos using natural language queries."""
    
    try:
        from .vector_db import ChromaVectorDB
        
        # Initialize vector database
        vector_db = ChromaVectorDB(
            collection_name=collection_name,
            persist_directory=str(Path(output_dir) / "vector_db")
        )
        vector_db.initialize()
        
        # Search
        results = vector_db.search(query, top_k=top_k, model_filter=model_filter)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        # Display results
        console.print(f"[bold blue]Video Search Results for: '{query}'[/bold blue]")
        console.print(f"Found {len(results)} matching video segments\n")
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            distance = result["distance"]
            
            console.print(f"[bold cyan]{i}. {result['id']}[/bold cyan]")
            console.print(f"   Model: {metadata.get('model_name', 'Unknown')}")
            console.print(f"   Video: {metadata.get('video_name', 'Unknown')}")
            console.print(f"   Shot: {metadata.get('shot_number', 'Unknown')}")
            console.print(f"   Time: {metadata.get('start_time', 0):.1f}s - {metadata.get('end_time', 0):.1f}s")
            console.print(f"   Distance: {distance:.3f}")
            console.print(f"   Content: {result['content'][:200]}...")
            console.print()
        
    except Exception as e:
        console.print(f"[red]✗ Search failed: {e}[/red]")
        logger.error(f"Search error: {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    collection_name: str = typer.Option("video_annotations", "--collection", "-c", help="Vector database collection name"),
    output_dir: str = typer.Option("./outputs", "--output", "-o", help="Output directory containing vector database"),
):
    """Display vector database statistics."""
    
    try:
        from .vector_db import ChromaVectorDB
        
        # Initialize vector database
        vector_db = ChromaVectorDB(
            collection_name=collection_name,
            persist_directory=str(Path(output_dir) / "vector_db")
        )
        vector_db.initialize()
        
        # Get statistics
        stats = vector_db.get_collection_stats()
        
        # Display statistics
        console.print(Panel.fit(
            f"[bold blue]Vector Database Statistics[/bold blue]\n"
            f"Collection: {stats['collection_name']}\n"
            f"Total Documents: {stats['total_documents']}\n"
            f"Models: {', '.join(stats['model_distribution'].keys())}\n"
            f"Videos: {', '.join(stats['video_distribution'].keys())}",
            title="Database Stats"
        ))
        
        # Model distribution table
        if stats['model_distribution']:
            table = Table(title="Model Distribution")
            table.add_column("Model", style="cyan")
            table.add_column("Documents", justify="right")
            
            for model, count in stats['model_distribution'].items():
                table.add_row(model, str(count))
            
            console.print(table)
        
        # Video distribution table
        if stats['video_distribution']:
            table = Table(title="Video Distribution")
            table.add_column("Video", style="cyan")
            table.add_column("Documents", justify="right")
            
            for video, count in stats['video_distribution'].items():
                table.add_row(video, str(count))
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Failed to get statistics: {e}[/red]")
        logger.error(f"Stats error: {e}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    save: Optional[str] = typer.Option(None, "--save", help="Save configuration to file"),
    load: Optional[str] = typer.Option(None, "--load", help="Load configuration from file"),
):
    """Manage configuration settings."""
    
    if show:
        settings = load_settings()
        console.print(Panel.fit(
            json.dumps(settings.to_dict(), indent=2),
            title="Current Configuration"
        ))
    
    elif save:
        settings = load_settings()
        save_settings(settings, save)
        console.print(f"[green]✓ Configuration saved to: {save}[/green]")
    
    elif load:
        if not Path(load).exists():
            console.print(f"[red]✗ Configuration file not found: {load}[/red]")
            raise typer.Exit(1)
        
        settings = load_settings(load)
        console.print(f"[green]✓ Configuration loaded from: {load}[/green]")
        console.print(Panel.fit(
            json.dumps(settings.to_dict(), indent=2),
            title="Loaded Configuration"
        ))
    
    else:
        console.print("[yellow]Please specify --show, --save, or --load[/yellow]")


@app.command()
def list_models():
    """List available models and their configurations."""
    
    settings = load_settings()
    
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Enabled", justify="center")
    table.add_column("Max Tokens", justify="right")
    table.add_column("Temperature", justify="right")
    
    for model in settings.models:
        table.add_row(
            model.name,
            model.model_id,
            "✓" if model.enabled else "✗",
            str(model.max_tokens),
            str(model.temperature)
        )
    
    console.print(table)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
