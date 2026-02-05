#!/usr/bin/env python3
"""
Test script for hierarchical generation with guided JSON.
Note: guided_json parameter is not supported in all vLLM versions.
"""

import os
import sys
import json

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

try:
    from vllm import LLM, SamplingParams
    print("vLLM imported successfully")
except Exception as e:
    print(f"Failed to import vLLM: {e}")
    sys.exit(1)

# Define JSON schema for hierarchy
schema = {
    "type": "object",
    "properties": {
        "level_1": {"type": "string"},
        "level_2": {"type": "string"},
        "level_3": {"type": "string"},
        "level_4": {"type": "string"},
        "level_5": {"type": "string"}
    },
    "required": ["level_1", "level_2", "level_3", "level_4", "level_5"]
}

# Initialize LLM
print("\nInitializing vLLM...")
try:
    llm = LLM(model="Qwen/Qwen3-0.6B", trust_remote_code=True, gpu_memory_utilization=0.9)
    print("vLLM initialized")
except Exception as e:
    print(f"Failed to initialize vLLM: {e}")
    sys.exit(1)

# Create prompt
metadata = {"object": "ceramic pot", "period": "Roman"}
prompt = f"""Generate a hierarchical caption with 5 levels of detail for this metadata: {json.dumps(metadata)}

Level 1: Single word
Level 2: Short phrase (2-4 words)
Level 3: Simple sentence
Level 4: Detailed sentence
Level 5: Comprehensive description

Output as JSON with keys level_1 through level_5."""

# Test with guided JSON
print("\nTesting with guided JSON schema...")
try:
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        guided_json=schema
    )
    
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text
    
    print(f"\nRaw output:\n{result}\n")
    
    # Parse JSON
    hierarchy = json.loads(result)
    print("Successfully parsed JSON")
    print("\nHierarchy:")
    for i in range(1, 6):
        print(f"  Level {i}: {hierarchy[f'level_{i}']}")
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
