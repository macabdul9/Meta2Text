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
        
        # Check if this is a Qwen2-VL model (requires special handling)
        self.is_qwen_vl = "qwen" in model_name.lower() and ("vl" in model_name.lower() or "vision" in model_name.lower())
        
        # Check if model is vision-language
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.is_vision = hasattr(self.processor, 'image_processor')
        except:
            self.processor = None
            self.is_vision = False
        
        # For Qwen2-VL models, use the recommended approach
        if self.is_qwen_vl:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                self.pipeline = None
                # Import qwen_vl_utils
                try:
                    from qwen_vl_utils import process_vision_info
                    self.process_vision_info = process_vision_info
                except ImportError:
                    raise ImportError("qwen-vl-utils package not installed. Install with: pip install qwen-vl-utils")
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen2-VL model: {e}")
        # For other vision models, load model directly for better control
        elif self.is_vision:
            try:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.pipeline = None  # Don't use pipeline for vision models
            except:
                # Fallback to pipeline if direct loading fails
                try:
                    from transformers import AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_name,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.pipeline = None
                except:
                    # Last resort: use pipeline
                    self.model = None
                    self.pipeline = pipeline(
                        "image-to-text",
                        model=model_name,
                        device_map="auto",
                        trust_remote_code=True
                    )
        else:
            self.model = None
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
        
        # Convert max_tokens to max_new_tokens (HuggingFace uses max_new_tokens)
        if 'max_tokens' in params:
            params['max_new_tokens'] = params.pop('max_tokens')
        
        # Valid generation parameters for HuggingFace models
        valid_gen_params = {
            'max_new_tokens', 'max_length', 'min_length', 'do_sample',
            'temperature', 'top_p', 'top_k', 'num_beams', 'num_return_sequences',
            'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size',
            'early_stopping', 'pad_token_id', 'eos_token_id', 'bos_token_id'
        }
        
        # Filter params to only include valid generation parameters
        gen_params = {k: v for k, v in params.items() if k in valid_gen_params}
        
        if self.is_vision and image:
            # Special handling for Qwen2-VL models
            if self.is_qwen_vl and self.processor and self.model:
                # For Qwen2-VL, we can pass image path or PIL Image directly
                # process_vision_info will handle both
                image_for_qwen = image
                if isinstance(image, str):
                    from pathlib import Path
                    if not Path(image).exists():
                        raise ValueError(f"Image path does not exist: {image}")
                    # Keep as path string for Qwen2-VL (process_vision_info handles paths)
                    image_for_qwen = image
                
                # Format messages according to Qwen2-VL requirements
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_for_qwen,  # Can be PIL Image or path
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                # Preparation for inference (Qwen2-VL recommended approach)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                # Move to same device as model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with filtered parameters
                generated_ids = self.model.generate(**inputs, **gen_params)
                
                # Trim input_ids from generated_ids
                input_ids = inputs['input_ids']
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
                # Decode
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                return output_text[0] if output_text else ""
            
            # Use processor and model directly for other vision models
            elif self.processor and self.model:
                # Handle image for other vision models
                if isinstance(image, str):
                    from pathlib import Path
                    if Path(image).exists():
                        image = Image.open(image)
                    else:
                        raise ValueError(f"Image path does not exist: {image}")
                
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                # Move to same device as model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with filtered parameters
                outputs = self.model.generate(**inputs, **gen_params)
                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Remove the input prompt from the generated text if present
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                return generated_text
            elif self.pipeline:
                # Fallback to pipeline
                result = self.pipeline(image, prompt=prompt, **gen_params)
                if isinstance(result, list):
                    return result[0].get('generated_text', str(result[0]))
                return str(result)
            else:
                raise ValueError("Neither model nor pipeline available for vision generation")
        else:
            # Text-only generation
            result = self.pipeline(prompt, **gen_params)
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
        
        # Check if vision model (would need special handling)
        self.is_vision = "vl" in model_name.lower() or "vision" in model_name.lower()
        
        # Initialize vLLM
        # Note: vLLM supports some vision models, but they need more memory
        # Use lower GPU memory utilization for vision models
        gpu_memory_util = 0.5 if self.is_vision else 0.9
        
        if self.is_vision:
            import warnings
            warnings.warn(
                f"vLLM vision model support may be limited for {model_name}. "
                "Consider using 'hf' engine for better vision model support."
            )
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_util
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
        
        # Handle image for vision models
        temp_file_path = None
        if self.is_vision and image:
            # vLLM vision models use multi_modal_data format
            if isinstance(image, str):
                from pathlib import Path
                if not Path(image).exists():
                    raise ValueError(f"Image path does not exist: {image}")
                image_path = image
            else:
                # PIL Image - save temporarily
                import tempfile
                import os
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                image.save(temp_file.name, format='JPEG')
                image_path = temp_file.name
                temp_file.close()
                temp_file_path = image_path
            
            # Format for vLLM vision models
            multimodal_prompt = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_path}
            }
            prompt_for_llm = multimodal_prompt
        else:
            if image:
                import warnings
                warnings.warn("vLLM engine: image input provided but model is not a vision model, using text-only generation")
            prompt_for_llm = prompt
        
        sampling_params = SamplingParams(
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            top_k=params.get('top_k', -1),
            max_tokens=params.get('max_tokens', params.get('max_new_tokens', 512)),
            stop=params.get('stop', None)
        )
        
        outputs = self.llm.generate([prompt_for_llm], sampling_params)
        result = outputs[0].outputs[0].text
        
        # Clean up temporary file if created
        if temp_file_path:
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return result


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
