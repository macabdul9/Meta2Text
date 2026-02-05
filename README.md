# Meta2Text: Archaeology Artifact Metadata to Text Description Pipeline

A flexible pipeline for converting archaeology artifact metadata into natural language text descriptions using various generation engines and methods.

## Features

- **Multiple Generation Engines**:
  - OpenAI API (GPT-4o-mini, GPT-4o, etc.)
  - HuggingFace Transformers pipeline
  - vLLM (for high-performance inference)

- **Six Generation Methods**:
  1. **Template-Based**: Rule-based template filling (zero hallucination)
  2. **LLM-Based Expansion**: Natural language generation from metadata
  3. **VLM-Assisted Hybrid**: Vision-language models with metadata grounding
  4. **Scene Graph Expansion**: Hierarchical/relational data conversion
  5. **Noise Injection**: VeCap approach for robust training
  6. **Hierarchical Generation**: Multi-level progressive detail (3-7 levels)

- **Flexible Input**:
  - Single instance (metadata + optional image)
  - HuggingFace datasets (batch processing)
  - Custom prompt templates

## Changelog

### Version 2.0 (February 2026)

**New Features:**
- **Hierarchical Caption Generation**: Generate captions at 3-7 levels of progressive detail (inspired by HierarCaps paper)
- **Custom Prompt Templates**: Support for user-defined prompts with template library in `prompts/` directory
- **Enhanced Multimodal Support**: Full vision-language model integration with image + metadata input

**Improvements:**
- Better vLLM compatibility and error handling
- Improved JSON parsing with fallback mechanisms
- Reorganized project structure for clarity

## Installation

### 1. Create Conda Environment

```bash
bash setup_conda.sh
```

Or manually:

```bash
conda create -n archaia python=3.10 -y
conda activate archaia
pip install -r requirements.txt
```

### 2. Set Environment Variables

For OpenAI engine:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For vLLM multiprocessing:
```bash
export VLLM_WORKER_MULTIPROC_METHOD='spawn'
```

## Usage

### Single Instance Mode

```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --generation_method llm_expansion \
    --output_path outputs/output.json
```

With image:
```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "ceramic pot", "period": "Roman"}' \
    --image_path assets/artifact.jpg \
    --generation_method vlm_hybrid
```

### HuggingFace Dataset Mode

```bash
python main.py \
    --generation_engine hf \
    --generation_model Qwen/Qwen2.5-4B-Instruct \
    --huggingface_dataset_path "username/dataset-name" \
    --image_column "image" \
    --metadata_column "metadata" \
    --generation_method llm_expansion \
    --output_path outputs/results.json
```

### Command Line Arguments

- `--generation_engine`: Engine type (`openai`, `vllm`, `hf`)
- `--generation_model`: Model name (e.g., `gpt-4o-mini`, `Qwen/Qwen2.5-4B-Instruct`)
- `--generation_params`: JSON string with generation parameters (default: `'{"temperature": 0.7, "max_tokens": 512}'`)
- `--image_path`: Path to single image file (optional)
- `--metadata`: JSON string or path to JSON file with metadata
- `--huggingface_dataset_path`: HuggingFace dataset path
- `--image_column`: Column name for images in dataset (required if dataset provided)
- `--metadata_column`: Column name for metadata in dataset (required if dataset provided)
- `--generation_method`: Method to use (`template`, `llm_expansion`, `vlm_hybrid`, `scene_graph`, `noise_injection`, `hierarchical`)
- `--num_levels`: Number of hierarchy levels for hierarchical generation (default: 5)
- `--hierarchy_direction`: Direction for hierarchy completion (default: `bidirectional`)
- `--prompt_template`: Path to custom prompt template file
- `--output_path`: Path to save output JSON file (optional, prints to stdout if not provided)
- `--batch_size`: Batch size for processing (default: 1)

## Generation Methods

### 1. Template-Based Generation
Fast, rule-based method with zero hallucination. Uses 50-100 varied templates.

```bash
--generation_method template
```

### 2. LLM-Based Expansion
Industry standard method using LLMs to create natural descriptions.

```bash
--generation_method llm_expansion
```

### 3. VLM-Assisted Hybrid Captioning
Uses vision-language models with metadata grounding. Requires image input.

```bash
--generation_method vlm_hybrid --image_path image.jpg
```

### 4. Scene Graph Expansion
Converts hierarchical/relational metadata to natural language.

```bash
--generation_method scene_graph
```

### 5. Noise Injection (VeCap)
Randomly swaps detailed captions with raw metadata strings.

```bash
--generation_method noise_injection
```

### 6. Hierarchical Caption Generation
Generates captions at 3-7 levels of granularity, from very general to very detailed. Inspired by the HierarCaps paper.

```bash
--generation_method hierarchical --num_levels 5
```

With custom prompt template:
```bash
--generation_method hierarchical \
--num_levels 5 \
--prompt_template prompts/hierarchical_archaeological.txt
```

Output format:
```json
{
  "hierarchy": {
    "level_1": "pot",
    "level_2": "ceramic pot",
    "level_3": "a Roman ceramic pot",
    "level_4": "an intact Roman ceramic pot with decorative patterns",
    "level_5": "a well-preserved intact Roman ceramic pot featuring intricate geometric decorative patterns, made of terracotta, dating from the 2nd century CE, with a reddish-brown surface and minor wear on the rim"
  }
}
```

## Examples

### Example 1: OpenAI with LLM Expansion

```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "bronze coin", "period": "Byzantine", "material": "bronze", "condition": "worn"}' \
    --generation_method llm_expansion \
    --generation_params '{"temperature": 0.8, "max_tokens": 256}'
```

### Example 2: HuggingFace Model with VLM Hybrid

```bash
python main.py \
    --generation_engine hf \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "amphora", "period": "Greek"}' \
    --image_path assets/artifact.jpg \
    --generation_method vlm_hybrid
```

### Example 3: Hierarchical Generation with vLLM

```bash
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --num_levels 5 \
    --output_path outputs/hierarchical.json
```

### Example 4: Custom Prompt Template

```bash
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-0.6B \
    --metadata '{"object": "statue", "material": "marble"}' \
    --generation_method hierarchical \
    --prompt_template prompts/hierarchical_archaeological.txt \
    --output_path outputs/custom_prompt.json
```

## Prompt Templates

The `prompts/` directory contains pre-built templates for different use cases:

- `hierarchical_default.txt` - Default 5-level hierarchy
- `hierarchical_3levels.txt` - Simplified 3-level hierarchy
- `hierarchical_7levels.txt` - Extended 7-level hierarchy
- `hierarchical_archaeological.txt` - Specialized for cultural heritage objects
- `llm_expansion.txt` - LLM expansion method
- `vlm_hybrid.txt` - Vision-language hybrid
- `scene_graph.txt` - Scene graph generation

Create custom templates using `{metadata}` and `{num_levels}` placeholders. See `prompts/README.md` for details.

## Output Format

The output is a JSON array with the following structure:

```json
[
  {
    "metadata": {
      "object": "ceramic pot",
      "period": "Roman"
    },
    "image_path": "assets/artifact.jpg",
    "description": "A well-preserved Roman ceramic pot..."
  }
]
```

For hierarchical generation:
```json
[
  {
    "metadata": {...},
    "image_path": "...",
    "hierarchy": {
      "level_1": "...",
      "level_2": "...",
      ...
    }
  }
]
```

## Notes

- For vision models, ensure you have appropriate GPU resources (8GB+ VRAM)
- vLLM engine may have limited vision model support; use `hf` engine for better vision compatibility
- Template-based method is fastest but least diverse
- VLM hybrid method requires image input
- Noise injection method randomly returns raw metadata (10% probability by default)
- Hierarchical generation works best with 5 levels; use 3 for simpler cases, 7 for maximum detail

## License

MIT
