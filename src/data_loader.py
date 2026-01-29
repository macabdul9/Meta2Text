"""Data loading utilities for single instances and HuggingFace datasets."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import os


class DataLoader:
    """Load data from various sources."""
    
    def load_single_instance(
        self,
        metadata: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a single instance with metadata and optional image.
        
        Args:
            metadata: Dictionary containing metadata
            image_path: Optional path to image file
            
        Returns:
            List with single item containing metadata and image_path
        """
        item = {
            'metadata': metadata,
            'image_path': image_path if image_path and Path(image_path).exists() else None
        }
        return [item]
    
    def load_hf_dataset(
        self,
        dataset_path: str,
        image_column: str,
        metadata_column: str,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Load data from HuggingFace dataset.
        
        Args:
            dataset_path: HuggingFace dataset path
            image_column: Name of column containing images
            metadata_column: Name of column containing metadata
            split: Dataset split to load
            
        Returns:
            List of items, each with metadata and image_path
        """
        try:
            dataset = load_dataset(dataset_path, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_path}: {e}")
        
        if image_column not in dataset.column_names:
            raise ValueError(f"Column '{image_column}' not found in dataset")
        if metadata_column not in dataset.column_names:
            raise ValueError(f"Column '{metadata_column}' not found in dataset")
        
        items = []
        for example in dataset:
            # Handle metadata - could be dict or JSON string
            metadata = example[metadata_column]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text and wrap in dict
                    metadata = {"text": metadata}
            
            # Handle image - could be PIL Image, path, or bytes
            image_path = None
            image_data = example[image_column]
            
            if image_data is None:
                image_path = None
            elif hasattr(image_data, 'save'):  # PIL Image
                # Keep PIL Image as-is, generation methods will handle it
                image_path = image_data
            elif isinstance(image_data, str):
                if Path(image_data).exists():
                    image_path = image_data
                else:
                    # Might be a URL or base64, handle in generation method
                    image_path = image_data
            else:
                # Could be bytes or other format
                image_path = image_data
            
            items.append({
                'metadata': metadata,
                'image_path': image_path
            })
        
        return items
