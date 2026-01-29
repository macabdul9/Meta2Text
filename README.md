# Meta2Text: Archaeology Artifact Metadata to Text Description Pipeline

A flexible pipeline for converting archaeology artifact metadata into natural language text descriptions using various generation engines and methods.

## Features

- **Multiple Generation Engines**:
  - OpenAI API (GPT-4o-mini, GPT-4o, etc.)
  - HuggingFace Transformers pipeline
  - vLLM (for high-performance inference)

- **Five Generation Methods**:
  1. **Template-Based**: Rule-based template filling (zero hallucination)
  2. **LLM-Based Expansion**: Natural language generation from metadata
  3. **VLM-Assisted Hybrid**: Vision-language models with metadata grounding
  4. **Scene Graph Expansion**: Hierarchical/relational data conversion
  5. **Noise Injection**: VeCap approach for robust training

- **Flexible Input**:
  - Single instance (metadata + optional image)
  - HuggingFace datasets (batch processing)

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

## Usage

### Single Instance Mode

```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --generation_method llm_expansion \
    --output_path output.json
```

With image:
```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "ceramic pot", "period": "Roman"}' \
    --image_path path/to/image.jpg \
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
    --output_path results.json
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
- `--generation_method`: Method to use (`template`, `llm_expansion`, `vlm_hybrid`, `scene_graph`, `noise_injection`)
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

### Example 3: Template-Based (Fast)

```bash
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "vase", "action": "displayed"}' \
    --generation_method template
```

## Output Format

The output is a JSON array with the following structure:

```json
[
  {
    "metadata": {
      "object": "ceramic pot",
      "period": "Roman"
    },
    "image_path": "path/to/image.jpg",
    "description": "A well-preserved Roman ceramic pot..."
  }
]
```

## Notes

- For vision models, ensure you have appropriate GPU resources
- vLLM engine may not fully support all vision models; use `hf` engine for vision models
- Template-based method is fastest but least diverse
- VLM hybrid method requires image input
- Noise injection method randomly returns raw metadata (10% probability by default)

## License

MIT
