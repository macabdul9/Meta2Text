# Prompt Templates

This directory contains prompt templates for different generation methods in Meta2Text.

## Available Templates

### General Generation Methods

- **`template_based.txt`** - Simple template-based generation
- **`llm_expansion.txt`** - LLM-based expansion of metadata
- **`vlm_hybrid.txt`** - Vision-language model with metadata
- **`scene_graph.txt`** - Scene graph structured descriptions

### Hierarchical Generation

- **`hierarchical_default.txt`** - Default 5-level hierarchical generation
- **`hierarchical_3levels.txt`** - Simplified 3-level hierarchy
- **`hierarchical_7levels.txt`** - Extended 7-level hierarchy
- **`hierarchical_archaeological.txt`** - Specialized for archaeological/cultural heritage objects

## Usage

Use the `--prompt_template` argument to specify a custom prompt:

```bash
python main.py \
    --generation_method hierarchical \
    --prompt_template prompts/hierarchical_7levels.txt \
    --metadata '{"object": "ceramic pot", "period": "Roman"}' \
    ...
```

## Creating Custom Templates

Templates use Python string formatting with the following placeholders:

- `{metadata}` - JSON-formatted metadata
- `{num_levels}` - Number of hierarchy levels (hierarchical methods only)

### Example Template

```
You are generating descriptions from metadata.

Metadata:
{metadata}

Generate a detailed description that incorporates all fields.
```

### Tips for Writing Prompts

1. Be specific about the desired output format
2. Provide clear instructions on what to include
3. Use examples when helpful
4. For hierarchical prompts, clearly define each level
5. Specify any domain-specific terminology or conventions
