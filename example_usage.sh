#!/bin/bash
# Example usage scripts for Meta2Text pipeline

echo "Example 1: Template-based generation (fastest)"
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --generation_method template

echo -e "\nExample 2: LLM-based expansion (most natural)"
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "bronze coin", "period": "Byzantine", "material": "bronze", "condition": "worn"}' \
    --generation_method llm_expansion \
    --generation_params '{"temperature": 0.8, "max_tokens": 256}'

echo -e "\nExample 3: VLM hybrid (requires image)"
# python main.py \
#     --generation_engine openai \
#     --generation_model gpt-4o-mini \
#     --metadata '{"object": "amphora", "period": "Greek"}' \
#     --image_path path/to/image.jpg \
#     --generation_method vlm_hybrid

echo -e "\nExample 4: Scene graph expansion"
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "vase", "relation": "contains", "target": "remains", "period": "Ancient Greek"}' \
    --generation_method scene_graph

echo -e "\nExample 5: Noise injection (VeCap)"
python main.py \
    --generation_engine openai \
    --generation_model gpt-4o-mini \
    --metadata '{"object": "artifact", "period": "Medieval"}' \
    --generation_method noise_injection
