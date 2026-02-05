#!/usr/bin/env python3
"""
Test script for vLLM with text and multimodal generation.
"""

import os
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM multimodal generation."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Test text generation
    print("Testing text generation...")
    text_llm = LLM(model="Qwen/Qwen3-0.6B")
    text_prompt = "The capital of France is"

    outputs = text_llm.generate([text_prompt], sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    del text_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Test multimodal generation
    print("\nTesting multimodal generation...")
    checkpoint_path = "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "assets/artifact.jpg"},
                {"type": "text", "text": "What do you see in this image?"},
            ],
        }
    ]

    inputs = prepare_inputs_for_vllm(messages, processor)

    multimodal_llm = LLM(
        model=checkpoint_path,
        gpu_memory_utilization=0.5
    )

    outputs = multimodal_llm.generate([inputs], sampling_params)
    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")