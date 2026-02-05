#!/usr/bin/env python3
"""
Multimodal hierarchical caption generation examples.

Demonstrates hierarchical caption generation using vision-language models
with both images and metadata.
"""

import os
import sys
import json
from pathlib import Path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from src.generation_engines import GenerationEngineFactory
from src.generation_methods import GenerationMethodFactory


def example_vllm_vision_model():
    """VLM-based hierarchical generation with image."""
    print("Example 1: Multimodal Hierarchical Generation with vLLM Vision Model")
    print("-" * 70)
    
    metadata = {
        "object": "ceramic pot",
        "period": "Roman",
        "condition": "intact",
        "material": "terracotta"
    }
    
    image_path = "assets/artifact.jpg"
    
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        print("Skipping this example.\n")
        return
    
    print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
    print(f"Image: {image_path}")
    
    print("\nInitializing Qwen3-VL-2B-Instruct...")
    try:
        engine = GenerationEngineFactory.create(
            engine_type="vllm",
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            generation_params={
                "temperature": 0.7,
                "max_tokens": 512
            }
        )
        print("Engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
    
    print("Initializing hierarchical generation method...")
    try:
        method = GenerationMethodFactory.create(
            method_type="hierarchical",
            engine=engine,
            num_levels=5
        )
        print("Method initialized successfully")
    except Exception as e:
        print(f"Failed to initialize method: {e}")
        return
    
    print("\nGenerating hierarchical captions...")
    try:
        hierarchy = method.generate(metadata=metadata, image_path=image_path)
        
        print("\nGenerated Hierarchy:")
        print("-" * 70)
        for i in range(1, 6):
            level_key = f"level_{i}"
            if level_key in hierarchy:
                print(f"Level {i}: {hierarchy[level_key]}")
        print("-" * 70)
        
        output = {
            "metadata": metadata,
            "image_path": image_path,
            "hierarchy": hierarchy
        }
        
        output_file = "outputs/multimodal_hierarchical.json"
        os.makedirs("outputs", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()


def example_huggingface_vlm():
    """HuggingFace VLM with hierarchical generation."""
    print("\n\nExample 2: Multimodal Hierarchical Generation with HuggingFace VLM")
    print("-" * 70)
    
    metadata = {
        "object": "bronze coin",
        "period": "Byzantine",
        "condition": "worn"
    }
    
    image_path = "assets/artifact.jpg"
    
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        print("Skipping this example.\n")
        return
    
    print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
    print(f"Image: {image_path}")
    
    print("\nInitializing HuggingFace vision model...")
    try:
        engine = GenerationEngineFactory.create(
            engine_type="hf",
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            generation_params={
                "temperature": 0.8,
                "max_tokens": 512
            }
        )
        print("Engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
    
    print("Initializing hierarchical generation method...")
    method = GenerationMethodFactory.create(
        method_type="hierarchical",
        engine=engine,
        num_levels=5
    )
    print("Method initialized successfully")
    
    print("\nGenerating hierarchical captions...")
    try:
        hierarchy = method.generate(metadata=metadata, image_path=image_path)
        
        print("\nGenerated Hierarchy:")
        print("-" * 70)
        for i in range(1, 6):
            level_key = f"level_{i}"
            if level_key in hierarchy:
                print(f"Level {i}: {hierarchy[level_key]}")
        print("-" * 70)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()


def example_vlm_hybrid_comparison():
    """VLM Hybrid method (non-hierarchical) for comparison."""
    print("\n\nExample 3: VLM Hybrid Method (for comparison)")
    print("-" * 70)
    
    metadata = {
        "object": "amphora",
        "period": "Greek",
        "function": "storage vessel"
    }
    
    image_path = "assets/artifact.jpg"
    
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        print("Skipping this example.\n")
        return
    
    print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
    print(f"Image: {image_path}")
    
    print("\nInitializing vLLM vision model...")
    try:
        engine = GenerationEngineFactory.create(
            engine_type="vllm",
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            generation_params={
                "temperature": 0.7,
                "max_tokens": 256
            }
        )
        print("Engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
    
    print("Initializing VLM hybrid method...")
    method = GenerationMethodFactory.create(
        method_type="vlm_hybrid",
        engine=engine
    )
    print("Method initialized successfully")
    
    print("\nGenerating single caption...")
    try:
        description = method.generate(metadata=metadata, image_path=image_path)
        
        print("\nGenerated Description:")
        print("-" * 70)
        print(description)
        print("-" * 70)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all multimodal examples."""
    print("\nMultimodal Hierarchical Caption Generation Examples")
    print("=" * 70)
    print("Demonstrating hierarchical caption generation using vision-language")
    print("models with both images and metadata.\n")
    
    example_vllm_vision_model()
    example_huggingface_vlm()
    example_vlm_hybrid_comparison()
    
    print("\n" + "=" * 70)
    print("Examples complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
