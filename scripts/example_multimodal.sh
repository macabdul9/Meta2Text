#!/bin/bash
# Multimodal hierarchical caption generation examples

export CUDA_VISIBLE_DEVICES=1

# Example 1: vLLM vision model with hierarchical generation
echo "Example 1: vLLM Vision Model with hierarchical generation"
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --num_levels 5 \
    --output_path outputs/multimodal_vllm.json

# Example 2: HuggingFace vision model with hierarchical generation
echo ""
echo "Example 2: HuggingFace Vision Model with hierarchical generation"
python main.py \
    --generation_engine hf \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "bronze coin", "period": "Byzantine"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --num_levels 5 \
    --output_path outputs/multimodal_hf.json

# Example 3: VLM hybrid method (single caption, for comparison)
echo ""
echo "Example 3: VLM Hybrid method (single caption)"
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "amphora", "period": "Greek"}' \
    --image_path assets/artifact.jpg \
    --generation_method vlm_hybrid \
    --output_path outputs/vlm_hybrid.json

# Example 4: Hierarchical with 3 levels
echo ""
echo "Example 4: Hierarchical with 3 levels"
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "statue", "material": "marble"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --num_levels 3 \
    --output_path outputs/multimodal_3levels.json

# Example 5: Using custom prompt template
echo ""
echo "Example 5: Using custom prompt template"
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "ceramic pot", "period": "Roman"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --prompt_template prompts/hierarchical_default.txt \
    --output_path outputs/custom_prompt.json

echo ""
echo "All examples complete"
