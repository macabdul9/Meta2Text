export CUDA_VISIBLE_DEVICES=0
python main.py \
    --generation_engine vllm \
    --generation_model Qwen/Qwen3-VL-2B-Instruct \
    --metadata '{"object": "ceramic pot", "period": "Roman", "condition": "intact"}' \
    --image_path assets/artifact.jpg \
    --generation_method hierarchical \
    --num_levels 5