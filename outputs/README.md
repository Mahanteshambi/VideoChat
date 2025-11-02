# Output Results

This directory contains example results from processing videos with the VideoIndex system.

## Files

### JSON Results

- `Hair Love_fastvlm_results.json` - FastVLM model results (25 shots)
- `Hair Love_qwen3_results.json` - Qwen3-VL model results (25 shots, fixed version)

These files contain structured metadata for each video shot, including:
- Shot descriptions
- Genre cues and themes
- Mood and setting context
- Processing times and success rates

### Comparison CSV Files

The `comparison/` directory contains side-by-side comparisons of different models:

- `fastvlm_vs_qwen3_25shots_fixed.csv` - Comparison between FastVLM and Qwen3 (25 shots)
- `fastvlm_vs_qwen3_25shots.csv` - Earlier comparison results
- `qwen2_vs_qwen3_comparison.csv` - Qwen2-VL vs Qwen3-VL comparison
- `qwen3_vs_fastvlm_comparison.csv` - Qwen3-VL vs FastVLM comparison

These CSV files include metrics such as:
- Processing time per shot
- Description quality
- Error rates
- Unique description counts

## Usage

You can use these results to:
- Understand the output format
- Compare model performance
- Analyze video understanding capabilities
- See example shot descriptions and metadata

## Note

The `vector_db/` directory is excluded from version control as it contains database files specific to the local environment.

