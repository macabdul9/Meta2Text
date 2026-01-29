"""Generation engines for different backends (OpenAI, HuggingFace, vLLM)."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from PIL import Image
import os


class GenerationEngine(ABC):
    """Base class for generation engines."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt and optional image.
        
        Args:
            prompt: Text prompt
            image: Optional image (path or PIL Image)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass


class OpenAIEngine(GenerationEngine):
    """OpenAI API-based generation engine."""
    
    def __init__(self, model_name: str, generation_params: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.generation_params = generation_params
    
    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> str:
        """Generate using OpenAI API."""
        params = {**self.generation_params, **kwargs}
        
        # Check if model supports vision
        vision_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-vision"]
        is_vision = any(vm in self.model_name.lower() for vm in vision_models)
        
        if image and is_vision:
            # Handle image for vision models
            from openai import OpenAI
            
            if isinstance(image, str):
                from pathlib import Path
                if Path(image).exists():
                    # Read image and encode
                    with open(image, 'rb') as f:
                        image_data = f.read()
                    import base64
                    import mimetypes
                    mime_type, _ = mimetypes.guess_type(image)
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                else:
                    # Assume it's already base64 or URL
                    image_data = image
                    base64_image = image
                    mime_type = "image/jpeg"
            else:
                # PIL Image
                import io
                import base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                mime_type = "image/png"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Filter out None values and prepare parameters
        filtered_params = {k: v for k, v in params.items() if v is not None}
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **filtered_params
        )
        
        return response.choices[0].message.content


class HuggingFaceEngine(GenerationEngine):
    """HuggingFace transformers pipeline-based generation engine."""
    
    def __init__(self, model_name: str, generation_params: Dict[str, Any]):
        try:
            from transformers import pipeline, AutoProcessor
            from PIL import Image
        except ImportError:
            raise ImportError("transformers package not installed. Install with: pip install transformers")
        
        self.model_name = model_name
        self.generation_params = generation_params
        
        # Check if model is vision-language
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.is_vision = hasattr(self.processor, 'image_processor')
        except:
            self.processor = None
            self.is_vision = False
        
        # Initialize pipeline
        if self.is_vision:
            self.pipeline = pipeline(
                "image-to-text" if "vl" in model_name.lower() or "vision" in model_name.lower() else "text-generation",
                model=model_name,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                trust_remote_code=True
            )
    
    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> str:
        """Generate using HuggingFace pipeline."""
        params = {**self.generation_params, **kwargs}
        
        if self.is_vision and image:
            # Handle image
            if isinstance(image, str):
                from pathlib import Path
                if Path(image).exists():
                    image = Image.open(image)
                else:
                    raise ValueError(f"Image path does not exist: {image}")
            
            # Use processor for vision models
            if self.processor:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                # Move to same device as model
                device = next(self.pipeline.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                outputs = self.pipeline.model.generate(**inputs, **params)
                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                return generated_text
            else:
                # Fallback to pipeline
                result = self.pipeline(image, prompt=prompt, **params)
                if isinstance(result, list):
                    return result[0].get('generated_text', str(result[0]))
                return str(result)
        else:
            # Text-only generation
            result = self.pipeline(prompt, **params)
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Remove prompt from generated text if it's included
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            return str(result)


class VLLMEngine(GenerationEngine):
    """vLLM-based generation engine."""
    
    def __init__(self, model_name: str, generation_params: Dict[str, Any]):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package not installed. Install with: pip install vllm")
        
        self.model_name = model_name
        self.generation_params = generation_params
        
        # Initialize vLLM
        # Note: vLLM doesn't natively support vision models, so we'll use it for text-only
        # For vision models, we might need to use a different approach
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        
        # Check if vision model (would need special handling)
        self.is_vision = "vl" in model_name.lower() or "vision" in model_name.lower()
        if self.is_vision:
            import warnings
            warnings.warn(
                f"vLLM may not fully support vision model {model_name}. "
                "Consider using 'hf' engine for vision models."
            )
    
    def generate(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> str:
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        params = {**self.generation_params, **kwargs}
        
        # vLLM doesn't handle images directly in the same way
        # For vision models, we'd need to preprocess the image and include it in the prompt
        # For now, we'll use text-only generation
        if image:
            import warnings
            warnings.warn("vLLM engine: image input is not fully supported, using text-only generation")
        
        sampling_params = SamplingParams(
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            top_k=params.get('top_k', -1),
            max_tokens=params.get('max_tokens', 512),
            stop=params.get('stop', None)
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


class GenerationEngineFactory:
    """Factory for creating generation engines."""
    
    @staticmethod
    def create(
        engine_type: str,
        model_name: str,
        generation_params: Dict[str, Any]
    ) -> GenerationEngine:
        """
        Create a generation engine.
        
        Args:
            engine_type: Type of engine ('openai', 'hf', 'vllm')
            model_name: Name of the model
            generation_params: Generation parameters
            
        Returns:
            GenerationEngine instance
        """
        if engine_type == "openai":
            return OpenAIEngine(model_name, generation_params)
        elif engine_type == "hf":
            return HuggingFaceEngine(model_name, generation_params)
        elif engine_type == "vllm":
            return VLLMEngine(model_name, generation_params)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
