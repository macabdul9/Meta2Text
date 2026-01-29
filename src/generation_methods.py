"""Generation methods for converting metadata to text descriptions."""

import random
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image


class GenerationMethod(ABC):
    """Base class for generation methods."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @abstractmethod
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """
        Generate text description from metadata and optional image.
        
        Args:
            metadata: Dictionary containing metadata
            image_path: Optional path to image file
            
        Returns:
            Generated text description
        """
        pass


class TemplateBasedMethod(GenerationMethod):
    """Template-based generation using rule-based templates."""
    
    def __init__(self, engine):
        super().__init__(engine)
        # Create a pool of varied templates
        self.templates = self._create_template_pool()
    
    def _create_template_pool(self) -> List[str]:
        """Create a pool of 50-100 varied templates."""
        templates = [
            "A photo of a {object} {action}.",
            "An image showing a {object} that is {action}.",
            "This photograph depicts a {object} {action}.",
            "A {object} can be seen {action} in this image.",
            "The image features a {object} {action}.",
            "Here is a {object} {action}.",
            "This picture shows a {object} {action}.",
            "A {object} appears {action} in the photograph.",
            "The photograph captures a {object} {action}.",
            "In this image, a {object} is {action}.",
            "A {object} {action} is visible in the photo.",
            "This is a photograph of a {object} {action}.",
            "The image presents a {object} {action}.",
            "A {object} {action} is shown here.",
            "This photo displays a {object} {action}.",
            "A {object} {action} can be observed in this image.",
            "The picture illustrates a {object} {action}.",
            "Here we see a {object} {action}.",
            "This image contains a {object} {action}.",
            "A {object} {action} is depicted in the photograph.",
            "The artifact {object} is shown {action}.",
            "An archaeological {object} appears {action}.",
            "This artifact, a {object}, is {action}.",
            "The {object} artifact is {action}.",
            "An ancient {object} is {action} in this image.",
            "The historical {object} is {action}.",
            "This {object} from the past is {action}.",
            "A preserved {object} is {action}.",
            "The {object} relic is {action}.",
            "An excavated {object} is {action}.",
            "This {object} artifact shows {action}.",
            "The {object} is {action} in the archaeological context.",
            "A {object} discovered at the site is {action}.",
            "The {object} specimen is {action}.",
            "This {object} find is {action}.",
            "An archaeological {object} is {action}.",
            "The {object} remains are {action}.",
            "A {object} from the excavation is {action}.",
            "This {object} piece is {action}.",
            "The {object} artifact displays {action}.",
            "A {object} from the collection is {action}.",
            "The {object} item is {action}.",
            "This {object} object is {action}.",
            "An ancient {object} artifact is {action}.",
            "The {object} discovery is {action}.",
            "A {object} from the dig is {action}.",
            "This {object} specimen shows {action}.",
            "The {object} find displays {action}.",
            "An archaeological {object} piece is {action}.",
            "The {object} relic appears {action}.",
            "A {object} from the site is {action}.",
            "This {object} artifact appears {action}.",
            "The {object} is {action} in the photograph.",
            "An excavated {object} is {action}.",
            "This {object} shows {action}.",
            "The {object} artifact is {action}.",
            "A {object} from the archaeological record is {action}.",
            "The {object} is {action} in this image.",
            "An ancient {object} is {action}.",
            "This {object} piece displays {action}.",
            "The {object} find is {action}.",
            "A {object} from the excavation site is {action}.",
            "The {object} specimen appears {action}.",
            "This {object} artifact is {action}.",
            "An archaeological {object} is {action}.",
            "The {object} relic is {action}.",
            "A {object} from the dig site is {action}.",
            "This {object} discovery is {action}.",
            "The {object} is {action} in the archaeological context.",
            "An ancient {object} artifact is {action}.",
            "The {object} remains are {action}.",
            "A {object} from the collection is {action}.",
            "This {object} item is {action}.",
            "The {object} object is {action}.",
            "An archaeological {object} is {action}.",
            "The {object} piece is {action}.",
            "A {object} from the site is {action}.",
            "This {object} find is {action}.",
            "The {object} artifact displays {action}.",
            "An excavated {object} is {action}.",
            "The {object} specimen is {action}.",
            "A {object} from the archaeological record is {action}.",
            "This {object} relic is {action}.",
            "The {object} is {action} in the photograph.",
            "An ancient {object} is {action}.",
            "This {object} piece shows {action}.",
            "The {object} discovery displays {action}.",
            "A {object} from the excavation is {action}.",
            "The {object} find appears {action}.",
            "An archaeological {object} artifact is {action}.",
            "This {object} is {action} in this image.",
            "The {object} remains are {action}.",
            "A {object} from the dig is {action}.",
            "The {object} item is {action}.",
            "An ancient {object} artifact is {action}.",
            "This {object} object is {action}.",
            "The {object} piece displays {action}.",
            "A {object} from the site is {action}.",
            "The {object} artifact is {action}.",
            "An excavated {object} is {action}.",
            "This {object} specimen shows {action}.",
            "The {object} relic is {action}.",
            "A {object} from the collection is {action}.",
            "The {object} find is {action}.",
            "An archaeological {object} is {action}.",
            "This {object} discovery is {action}.",
            "The {object} is {action} in the archaeological context."
        ]
        return templates
    
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """Generate using template-based method."""
        # Randomly sample a template
        template = random.choice(self.templates)
        
        # Extract common fields from metadata
        object_name = metadata.get('object', metadata.get('name', metadata.get('artifact', 'artifact')))
        action = metadata.get('action', metadata.get('state', metadata.get('condition', 'shown')))
        
        # Fill template
        try:
            description = template.format(object=object_name, action=action)
        except KeyError:
            # If template has other placeholders, try to fill from metadata
            description = template
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    description = description.replace(f"{{{key}}}", str(value))
        
        return description


class LLMExpansionMethod(GenerationMethod):
    """LLM-based expansion of metadata to natural language."""
    
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """Generate using LLM expansion."""
        # Create prompt for LLM
        metadata_str = json.dumps(metadata, indent=2)
        prompt = f"""You are a professional caption writer for archaeology artifacts. 
Create a detailed, natural language description based on the following metadata.

Metadata:
{metadata_str}

Instructions:
- Create a natural, engaging description
- Include all key information from the metadata
- Use varied vocabulary and sentence structures
- Infer reasonable details about lighting, texture, and condition if not specified
- Write in a professional but accessible tone
- Do not invent details that are not suggested by the metadata

Description:"""
        
        # Generate using engine (text-only, no image)
        description = self.engine.generate(prompt=prompt, image=None)
        return description.strip()


class VLMHybridMethod(GenerationMethod):
    """VLM-assisted hybrid captioning using image and metadata."""
    
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """Generate using VLM hybrid method."""
        if not image_path:
            # Fallback to LLM expansion if no image
            llm_method = LLMExpansionMethod(self.engine)
            return llm_method.generate(metadata, image_path)
        
        # Load image
        if isinstance(image_path, str):
            if Path(image_path).exists():
                image = Image.open(image_path)
            else:
                # Might be a URL or already loaded, pass as-is
                image = image_path
        else:
            image = image_path
        
        # Create prompt that includes metadata requirements
        metadata_str = json.dumps(metadata, indent=2)
        metadata_terms = ", ".join([f"{k}: {v}" for k, v in metadata.items() if isinstance(v, (str, int, float))])
        
        prompt = f"""Describe this image in detail. You must include these terms and concepts: {metadata_terms}

Additional metadata context:
{metadata_str}

Create a natural, detailed description that:
- Accurately describes what you see in the image
- Incorporates the required metadata terms naturally
- Includes observations about lighting, texture, condition, and visual details
- Uses professional archaeological terminology when appropriate

Description:"""
        
        # Generate using engine with image
        description = self.engine.generate(prompt=prompt, image=image)
        return description.strip()


class SceneGraphMethod(GenerationMethod):
    """Scene graph expansion for hierarchical/relational data."""
    
    def _build_scene_graph(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata to scene graph structure."""
        graph = {
            "objects": [],
            "relations": [],
            "attributes": []
        }
        
        # Extract objects
        if "object" in metadata or "name" in metadata:
            obj_name = metadata.get("object") or metadata.get("name")
            graph["objects"].append({"name": obj_name, "id": 0})
        
        # Extract relations
        if "relation" in metadata:
            graph["relations"].append(metadata["relation"])
        
        # Extract attributes
        for key, value in metadata.items():
            if key not in ["object", "name", "relation"]:
                graph["attributes"].append({"key": key, "value": value})
        
        return graph
    
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """Generate using scene graph method."""
        # Build scene graph
        graph = self._build_scene_graph(metadata)
        
        # Convert graph to text prompt
        graph_str = json.dumps(graph, indent=2)
        prompt = f"""Convert the following scene graph representation into a natural language description.

Scene Graph:
{graph_str}

Instructions:
- Convert the graph structure into fluent, natural sentences
- Describe objects, their attributes, and relationships
- Use varied sentence structures
- Create a coherent narrative from the graph

Description:"""
        
        # Generate using engine
        description = self.engine.generate(prompt=prompt, image=image_path)
        return description.strip()


class NoiseInjectionMethod(GenerationMethod):
    """Noise injection method (VeCap approach)."""
    
    def __init__(self, engine, noise_probability: float = 0.1):
        super().__init__(engine)
        self.noise_probability = noise_probability
        self.base_method = LLMExpansionMethod(engine)
    
    def _metadata_to_string(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata to raw string format."""
        parts = []
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
        return ", ".join(parts)
    
    def generate(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> str:
        """Generate using noise injection method."""
        # Randomly decide whether to inject noise
        if random.random() < self.noise_probability:
            # Return raw metadata string
            return self._metadata_to_string(metadata)
        else:
            # Generate detailed caption using base method
            return self.base_method.generate(metadata, image_path)


class GenerationMethodFactory:
    """Factory for creating generation methods."""
    
    @staticmethod
    def create(method_type: str, engine) -> GenerationMethod:
        """
        Create a generation method.
        
        Args:
            method_type: Type of method ('template', 'llm_expansion', 'vlm_hybrid', 'scene_graph', 'noise_injection')
            engine: GenerationEngine instance
            
        Returns:
            GenerationMethod instance
        """
        if method_type == "template":
            return TemplateBasedMethod(engine)
        elif method_type == "llm_expansion":
            return LLMExpansionMethod(engine)
        elif method_type == "vlm_hybrid":
            return VLMHybridMethod(engine)
        elif method_type == "scene_graph":
            return SceneGraphMethod(engine)
        elif method_type == "noise_injection":
            return NoiseInjectionMethod(engine)
        else:
            raise ValueError(f"Unknown method type: {method_type}")
