from vllm import LLM, SamplingParams
import torch
import gc

# Sampling parameters -- shared across all models
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Text generation model
text_llm = LLM(model="Qwen/Qwen3-0.6B")

text_prompt = "The capital of France is"

print("Testing text generation:")
outputs = text_llm.generate([text_prompt], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")

# Delete the text generation model to free up memory
del text_llm
torch.cuda.empty_cache()
gc.collect()

# Image + text generation model
# Use lower GPU memory utilization for vision models as they need more memory
multimodal_llm = LLM(
    model="Qwen/Qwen3-VL-2B-Instruct",
    gpu_memory_utilization=0.5  # Lower utilization to leave room for image processing
)

image_prompt = {
    "prompt": "What do you see in this image?",
    "multi_modal_data": {"image": "assets/artifact.jpg"}
}

print("\nTesting image + text generation:")
outputs = multimodal_llm.generate([image_prompt], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")