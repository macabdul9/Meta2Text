#!/usr/bin/env python3
"""
Test script for hierarchical caption generation using vLLM.
"""

import os
import sys
import json

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from src.generation_engines import GenerationEngineFactory
from src.generation_methods import GenerationMethodFactory


def test_hierarchical_generation():
    """Test hierarchical generation with vLLM."""
    
    print("Testing Hierarchical Caption Generation")
    print("-" * 60)
    
    # Test metadata
    test_cases = [
        {
            "object": "ceramic pot",
            "period": "Roman",
            "condition": "intact",
            "material": "terracotta",
            "decoration": "geometric patterns"
        },
        {
            "object": "bronze coin",
            "period": "Byzantine",
            "condition": "worn",
            "material": "bronze"
        },
        {
            "object": "amphora",
            "period": "Greek",
            "condition": "fragmentary",
            "material": "clay",
            "function": "storage vessel"
        }
    ]
    
    # Initialize vLLM engine
    print("\n1. Initializing vLLM engine...")
    try:
        engine = GenerationEngineFactory.create(
            engine_type="vllm",
            model_name="Qwen/Qwen3-0.6B",
            generation_params={
                "temperature": 0.8,
                "max_tokens": 512
            }
        )
        print("   Engine initialized")
    except Exception as e:
        print(f"   Failed to initialize engine: {e}")
        return
    
    # Initialize hierarchical generation method
    print("\n2. Initializing hierarchical generation method...")
    try:
        method = GenerationMethodFactory.create(
            method_type="hierarchical",
            engine=engine,
            num_levels=5,
            direction="bidirectional"
        )
        print("   Method initialized")
    except Exception as e:
        print(f"   Failed to initialize method: {e}")
        return
    
    # Test generation for each case
    print("\n3. Generating hierarchical captions...")
    results = []
    
    for i, metadata in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}:")
        print(f"   Metadata: {json.dumps(metadata, indent=6)}")
        
        try:
            hierarchy = method.generate(metadata=metadata, image_path=None)
            
            print(f"   Generated Hierarchy:")
            for level in range(1, 6):
                level_key = f"level_{level}"
                if level_key in hierarchy:
                    print(f"     Level {level}: {hierarchy[level_key]}")
            
            results.append({
                "metadata": metadata,
                "hierarchy": hierarchy
            })
            
            print(f"   Generation successful")
            
        except Exception as e:
            print(f"   Generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_path = "outputs/test_hierarchical_output.json"
    print(f"\n4. Saving results to {output_path}...")
    try:
        os.makedirs("outputs", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved")
    except Exception as e:
        print(f"   Failed to save results: {e}")
    
    print("\n" + "-" * 60)
    print("Test complete")
    print("-" * 60)


if __name__ == "__main__":
    test_hierarchical_generation()
