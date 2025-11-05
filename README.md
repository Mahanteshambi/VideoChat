# VideoIndex

A modular Python project for video indexing and search using multiple vision-language models (FastVLM, Qwen2-VL, Qwen3-VL) with vector database storage.

## Features

- **Video Indexing**: Extract and index video content using state-of-the-art vision-language models
- **Multi-Model Support**: Compare FastVLM, Qwen2-VL, and Qwen3-VL models for comprehensive video understanding
- **Semantic Search**: Search through indexed videos using natural language queries
- **Vector Database Integration**: Store and retrieve video annotations using ChromaDB
- **Modular Architecture**: Clean, extensible codebase with proper separation of concerns
- **Rich CLI Interface**: Beautiful command-line interface with progress tracking
- **Flexible Configuration**: Environment variables and config file support
- **Shot Detection**: Automatic video shot detection and processing
- **JSON Export**: Export results in structured JSON format

## Installation

This project uses `uv` as the package manager. Install dependencies with:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or install in development mode
uv sync --dev
```

## Quick Start

### Process a Video

```bash
# Index a video with all available models
uv run videoindex process path/to/your/video.mp4

# Index with specific models only
uv run videoindex process path/to/your/video.mp4 --models FastVLM,Qwen3

# Limit number of shots for testing
uv run videoindex process path/to/your/video.mp4 --max-shots 25

# Resize frames before processing (faster, lower quality)
uv run videoindex process path/to/your/video.mp4 --max-frame-side 512

# Custom output directory
uv run videoindex process path/to/your/video.mp4 --output ./my_results

# Skip vector database storage
uv run videoindex process path/to/your/video.mp4 --no-vector-db

# Skip JSON file output
uv run videoindex process path/to/your/video.mp4 --no-json

# Use custom configuration file
uv run videoindex process path/to/your/video.mp4 --config config.json

# Use custom vector database collection
uv run videoindex process path/to/your/video.mp4 --collection my_collection
```

### Search Indexed Videos

```bash
# Search for specific content
uv run videoindex search "action scene with car chase"

# Filter results by model
uv run videoindex search "romantic scene" --model FastVLM

# Get more results
uv run videoindex search "outdoor landscape" --top-k 20

# Use custom output directory (where vector DB is stored)
uv run videoindex search "query" --output ./my_results

# Use custom collection
uv run videoindex search "query" --collection my_collection
```

### View Statistics

```bash
# Show database statistics
uv run videoindex stats

# Show statistics for custom collection
uv run videoindex stats --collection my_annotations

# Use custom output directory
uv run videoindex stats --output ./my_results
```

### Configuration Management

```bash
# Show current configuration
uv run videoindex config --show

# Save configuration to file
uv run videoindex config --save config.json

# Load configuration from file
uv run videoindex config --load config.json
```

### List Available Models

```bash
# List all available models and their settings
uv run videoindex list-models
```

### Compare Models (Python Examples)

The repository includes example scripts for comparing models:

```bash
# Basic usage example
python examples/basic_usage.py

# Compare multiple models on the same video
python examples/compare_models.py path/to/video.mp4

# Compare specific models
python examples/compare_models.py path/to/video.mp4 FastVLM,Qwen3
```

These examples demonstrate how to use the Python API directly and compare model performance.

## Configuration

### Environment Variables

- `VIDEOINDEX_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `VIDEOINDEX_DEVICE`: Device to use (cuda, mps, cpu, or auto)
- `VIDEOINDEX_OUTPUT_DIR`: Default output directory
- `VIDEOINDEX_VECTOR_DB_COLLECTION`: Default collection name

### Configuration File

Create a `config.json` file to customize settings:

```json
{
  "models": [
    {
      "name": "FastVLM",
      "model_id": "apple/FastVLM-0.5B",
      "enabled": true,
      "max_tokens": 512,
      "temperature": 0.1,
      "top_p": 0.9
    },
    {
      "name": "Quen2",
      "model_id": "Qwen/Qwen2-VL-2B-Instruct",
      "enabled": true,
      "max_tokens": 512,
      "temperature": 0.1,
      "top_p": 0.9
    },
    {
      "name": "Qwen3",
      "model_id": "Qwen/Qwen3-VL-2B-Instruct",
      "enabled": true,
      "max_tokens": 512,
      "temperature": 0.3,
      "top_p": 0.85
    }
  ],
  "vector_db": {
    "enabled": true,
    "collection_name": "video_annotations",
    "persist_directory": "./chroma_db",
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "processing": {
    "output_directory": "./outputs",
    "save_json": true,
    "save_to_vector_db": true,
    "compare_models": true,
    "max_frames_per_shot": 5,
    "shot_detection_threshold": 27
  },
  "log_level": "INFO"
}
```

## Output Files

Results are saved to the output directory (default: `./outputs`):

- `{video_name}_{model_name}_results.json` - Individual model results
- `{video_name}_combined_results.json` - Combined results when comparing models
- `vector_db/` - ChromaDB database files for semantic search
- `comparison/` - Comparison CSV files (if generated separately)

### JSON Results Format

Each model produces a JSON file with the following structure:

```json
[
  {
    "shot_info": {
      "shot_number": 1,
      "start_time_seconds": 0.0,
      "end_time_seconds": 5.2,
      "duration_seconds": 5.2
    },
    "metadata": {
      "ShotDescription": "A person walking down a busy street...",
      "GenreCues": [
        {
          "genre_hint": "drama",
          "prominence_in_shot": 0.8
        }
      ],
      "SubgenreCues": ["slice of life"],
      "AdjectiveTheme": ["urban", "busy"],
      "Mood": ["energetic", "urban"],
      "SettingContext": ["city street", "daytime"],
      "ContentDescriptors": ["person", "walking", "crowd"],
      "LocationHints_Regional": ["urban area"],
      "LocationHints_International": ["city"],
      "SearchKeywords": ["walking", "street", "urban", "person"]
    },
    "model_name": "FastVLM",
    "processing_time_seconds": 2.34,
    "success": true,
    "error_message": null
  }
]
```

### Vector Database

Annotations are stored in ChromaDB with:
- **Content**: Searchable text combining all metadata fields
- **Metadata**: Structured information about the shot and processing
- **Embeddings**: Vector representations for semantic search

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run isort src/
```

### Type Checking

```bash
uv run mypy src/
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional, for GPU acceleration)
- Sufficient disk space for model downloads and vector database

## Command Reference

### `process` - Index Videos

**Usage:** `uv run videoindex process <video_path> [OPTIONS]`

**Arguments:**
- `video_path` - Path to the video file to index

**Options:**
- `--output, -o` - Output directory (default: `./outputs`)
- `--models, -m` - Comma-separated list of models (FastVLM, Quen2, Qwen3)
- `--max-shots` - Maximum number of shots to process
- `--max-frame-side` - Resize frames so max(width,height)=N before inference
- `--no-vector-db` - Skip saving to vector database
- `--no-json` - Skip saving JSON results
- `--collection, -c` - Vector database collection name (default: `video_annotations`)
- `--config` - Path to configuration file

### `search` - Search Indexed Videos

**Usage:** `uv run videoindex search <query> [OPTIONS]`

**Arguments:**
- `query` - Natural language search query

**Options:**
- `--output, -o` - Output directory containing vector database (default: `./outputs`)
- `--collection, -c` - Vector database collection name (default: `video_annotations`)
- `--model` - Filter results by model name
- `--top-k` - Number of results to return (default: 10)

### `stats` - Database Statistics

**Usage:** `uv run videoindex stats [OPTIONS]`

**Options:**
- `--output, -o` - Output directory containing vector database (default: `./outputs`)
- `--collection, -c` - Vector database collection name (default: `video_annotations`)

### `config` - Configuration Management

**Usage:** `uv run videoindex config [OPTIONS]`

**Options:**
- `--show` - Show current configuration
- `--save <file>` - Save configuration to file
- `--load <file>` - Load configuration from file

### `list-models` - List Available Models

**Usage:** `uv run videoindex list-models`

Displays all available models with their configuration (enabled status, max tokens, temperature, etc.)

## Model Information

### FastVLM
- **Model**: `apple/FastVLM-0.5B`
- **Size**: ~0.5B parameters
- **Specialization**: Fast vision-language understanding
- **Best for**: Quick processing, good general performance
- **Speed**: ~22s per shot
- **Quality**: 80% real descriptions (some placeholder text)

### Qwen2-VL
- **Model**: `Qwen/Qwen2-VL-2B-Instruct`
- **Size**: ~2B parameters
- **Specialization**: Advanced vision-language reasoning
- **Best for**: Detailed analysis, complex scene understanding
- **Speed**: ~97s per shot
- **Quality**: Can have parsing issues with markdown responses

### Qwen3-VL
- **Model**: `Qwen/Qwen3-VL-2B-Instruct`
- **Size**: ~2B parameters
- **Specialization**: Improved video understanding with numerical stability
- **Best for**: Production use with high reliability
- **Speed**: ~63s per shot
- **Quality**: 100% real descriptions, excellent consistency

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.
