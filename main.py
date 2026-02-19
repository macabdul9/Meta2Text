#!/usr/bin/env python3
"""
Main CLI entry point for Meta2Text pipeline.
Converts archaeology artifact metadata into text descriptions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from src.generation_engines import GenerationEngineFactory
from src.generation_methods import GenerationMethodFactory
from src.data_loader import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert archaeology artifact metadata to text descriptions"
    )
    
    # Generation engine
    parser.add_argument(
        "--generation_engine",
        type=str,
        choices=["openai", "vllm", "hf"],
        default="openai",
        help="Generation engine to use (default: openai)"
    )
    
    # Generation model
    parser.add_argument(
        "--generation_model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini). Examples: gpt-4o-mini, Qwen/Qwen2.5-4B-Instruct, Qwen/Qwen2-VL-2B-Instruct"
    )
    
    # Generation parameters
    parser.add_argument(
        "--generation_params",
        type=str,
        default='{"temperature": 0.7, "max_tokens": 512}',
        help="JSON string with generation parameters (default: '{\"temperature\": 0.7, \"max_tokens\": 512}')"
    )
    
    # Input options
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to single image file (optional)"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="JSON string or path to JSON file with metadata (optional)"
    )
    
    # HuggingFace dataset options
    parser.add_argument(
        "--huggingface_dataset_path",
        type=str,
        default=None,
        help="HuggingFace dataset path (required if image_path and metadata are None)"
    )
    
    parser.add_argument(
        "--image_column",
        type=str,
        default=None,
        help="Column name for images in HuggingFace dataset (required if dataset is provided)"
    )
    
    parser.add_argument(
        "--metadata_column",
        type=str,
        default=None,
        help="Column name for metadata in HuggingFace dataset (required if dataset is provided)"
    )
    
    # Generation method
    parser.add_argument(
        "--generation_method",
        type=str,
        choices=["template", "llm_expansion", "vlm_hybrid", "scene_graph", "noise_injection", "hierarchical"],
        default="llm_expansion",
        help="Generation method to use (default: llm_expansion)"
    )
    
    # Hierarchical generation options
    parser.add_argument(
        "--num_levels",
        type=int,
        default=5,
        help="Number of hierarchy levels for hierarchical generation (default: 5)"
    )
    
    parser.add_argument(
        "--hierarchy_direction",
        type=str,
        choices=["forward", "backward", "bidirectional"],
        default="bidirectional",
        help="Direction for hierarchy completion (default: bidirectional)"
    )
    
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Path to custom prompt template file for hierarchical generation"
    )
    
    # Output options
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON file (optional, prints to stdout if not provided)"
    )
    
    # Additional options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check if dataset is provided
    if args.huggingface_dataset_path:
        if not args.image_column:
            raise ValueError("--image_column is required when --huggingface_dataset_path is provided")
        if not args.metadata_column:
            raise ValueError("--metadata_column is required when --huggingface_dataset_path is provided")
    
    # Check if single instance mode
    if not args.huggingface_dataset_path:
        if not args.metadata:
            raise ValueError("Either --metadata or --huggingface_dataset_path must be provided")
    
    # Parse generation params
    try:
        if isinstance(args.generation_params, str):
            args.generation_params = json.loads(args.generation_params)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in --generation_params: {args.generation_params}")
    
    # Parse metadata if it's a string
    if args.metadata:
        # Check if it's a file path
        metadata_path = Path(args.metadata)
        try:
            if metadata_path.exists() and metadata_path.is_file():
                with open(metadata_path, 'r') as f:
                    args.metadata = json.load(f)
            else:
                # Try to parse as JSON string
                try:
                    args.metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in --metadata: {args.metadata}")
        except OSError:
            # If the path is too long (e.g., it's actually a JSON string), treat as JSON
            try:
                args.metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in --metadata: {args.metadata}")
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        args = validate_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize generation engine
    try:
        engine = GenerationEngineFactory.create(
            engine_type=args.generation_engine,
            model_name=args.generation_model,
            generation_params=args.generation_params
        )
    except Exception as e:
        print(f"Error initializing generation engine: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize generation method
    try:
        method_kwargs = {}
        if args.generation_method == "hierarchical":
            method_kwargs['num_levels'] = args.num_levels
            method_kwargs['direction'] = args.hierarchy_direction
            method_kwargs['prompt_template'] = args.prompt_template
        
        method = GenerationMethodFactory.create(
            method_type=args.generation_method,
            engine=engine,
            **method_kwargs
        )
    except Exception as e:
        print(f"Error initializing generation method: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    try:
        data_loader = DataLoader()
        if args.huggingface_dataset_path:
            data = data_loader.load_hf_dataset(
                dataset_path=args.huggingface_dataset_path,
                image_column=args.image_column,
                metadata_column=args.metadata_column
            )
        else:
            data = data_loader.load_single_instance(
                metadata=args.metadata,
                image_path=args.image_path
            )
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate descriptions
    try:
        results = []
        for item in data:
            description = method.generate(
                metadata=item['metadata'],
                image_path=item.get('image_path')
            )
            
            # Handle hierarchical output (dict) vs regular output (str)
            result_item = {
                'metadata': item['metadata'],
                'image_path': item.get('image_path')
            }
            
            if isinstance(description, dict):
                # Hierarchical output
                result_item['hierarchy'] = description
            else:
                # Regular string output
                result_item['description'] = description
            
            results.append(result_item)
        
        # Output results
        if args.output_path:
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_path}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Error generating descriptions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
