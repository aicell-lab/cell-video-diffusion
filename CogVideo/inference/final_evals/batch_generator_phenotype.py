"""
Batch generator for CogVideoX evaluations with phenotype control.
This script loads the model once and generates multiple phenotype-controlled videos.
Based on cli_demo2.py functionality.
"""

import argparse
import logging
import os
import json
import torch
import sys
from typing import List, Optional, Dict, Any, Literal

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import export_to_video

# Add the parent directory to sys.path to import from finetune
sys.path.append("../..")
from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder
from finetune.models.modules.combined_model import CombinedTransformerWithEmbedder

from safetensors import safe_open

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
    # For SFT models converted to fp32
    "fp32_model": (768, 1360),
}

def load_sft_combined_model(
    base_dir: str,
    device: torch.device = torch.device("cuda"),
    phenotype_module: str = "single",
    phenotype_dim: int = 4,
    phenotype_hidden_dim: int = 256,
    phenotype_output_dim: int = 4096,
    phenotype_dropout: float = 0.1,
    strict_load: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> CombinedTransformerWithEmbedder:
    """
    Loads a CogVideoXTransformer3DModel + PhenotypeEmbedder from FP32 safetensors shards
    in the given base_dir, merges them into a single state_dict, and returns a
    CombinedTransformerWithEmbedder. Moves the model to `device`.
    """
    # 1. Load config
    config_file = os.path.join(base_dir, "config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # 2. Instantiate base CogVideoXTransformer3DModel from config
    transformer = CogVideoXTransformer3DModel(**config_dict)
    logging.info(f"Instantiated CogVideoXTransformer3DModel from {config_file}")

    # 3. Instantiate the phenotype embedder
    phenotype_embedder = PhenotypeEmbedder(
        input_dim=phenotype_dim,
        hidden_dim=phenotype_hidden_dim,
        output_dim=phenotype_output_dim,
        dropout=phenotype_dropout,
    )
    logging.info(f"Instantiated PhenotypeEmbedder")

    # 4. Wrap them in the combined model
    combined_model = CombinedTransformerWithEmbedder(
        transformer=transformer,
        phenotype_embedder=phenotype_embedder,
        phenotype_module=phenotype_module,
    )
    logging.info(f"Created CombinedTransformerWithEmbedder")

    # 5. Merge safetensor shards into a single state_dict
    logging.info(f"Loading safetensors from {base_dir}")
    state_dict = {}
    num_shards = 5
    for idx in range(1, num_shards + 1):
        shard_file = os.path.join(base_dir, f"model-0000{idx}-of-00005.safetensors")
        if not os.path.exists(shard_file):
            raise FileNotFoundError(f"Missing shard file: {shard_file}")
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # 6. Load the state_dict
    missing_keys, unexpected_keys = combined_model.load_state_dict(state_dict, strict=strict_load)
    logging.info(f"Missing keys: {missing_keys}")
    logging.info(f"Unexpected keys: {unexpected_keys}")

    # 7. Eval + move to device
    combined_model.eval()
    combined_model.to(device, dtype=dtype)
    logging.info(f"Model loaded and moved to {device}")

    return combined_model

def setup_pipe(
    model_path: str,
    sft_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = "t2v",
):
    """Set up the CogVideoX pipeline with the specified model."""
    logging.info(f"Loading model from {model_path}")
    
    # Determine model resolution
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, RESOLUTION_MAP["fp32_model"])
    
    # Load the base pipeline
    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Generation type {generate_type} not supported for phenotype control")
    
    # Load SFT model if specified
    if sft_path:
        logging.info(f"Loading SFT combined model from {sft_path}")
        # This returns a CombinedTransformerWithEmbedder
        combined_model = load_sft_combined_model(
            base_dir=sft_path,
            device=torch.device("cuda"),
            phenotype_module="single",  # or "multi", whichever you used
            phenotype_dim=4,           # match your training
            phenotype_hidden_dim=256,
            phenotype_output_dim=4096,
            phenotype_dropout=0.1,
            strict_load=False,
            dtype=dtype
        )
        # Replace the transformer in the pipeline
        pipe.transformer = combined_model
        logging.info(f"Successfully replaced pipeline.transformer with CombinedTransformerWithEmbedder from SFT!")
    
    # Load LoRA weights if specified
    if lora_path:
        logging.info(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1)
    
    # Set up scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    # Make sure the device is set before calling enable_sequential_cpu_offload
    if not hasattr(pipe, 'device'):
        pipe.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable CPU offload for memory efficiency
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe, desired_resolution

def generate_videos_from_config(config_file: str):
    """Generate multiple videos based on a configuration file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Setup common parameters
    model_path = config.get("model_path", "../../models/CogVideoX1.5-5B")
    sft_path = config.get("sft_path", None)
    lora_path = config.get("lora_path", None)
    lora_rank = config.get("lora_rank", 128)
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    generate_type = config.get("generate_type", "t2v")
    
    # Load the model once
    pipe, default_resolution = setup_pipe(model_path, sft_path, lora_path, lora_rank, dtype, generate_type)
    
    # Process each generation task
    for task in config.get("tasks", []):
        prompt = task.get("prompt", "Time-lapse microscopy video of cells.")
        output_dir = task.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        seeds = task.get("seeds", [42])
        width = task.get("width", default_resolution[1])
        height = task.get("height", default_resolution[0])
        num_frames = task.get("num_frames", 81)
        fps = task.get("fps", 10)
        guidance_scale = task.get("guidance_scale", 8.0)
        num_inference_steps = task.get("num_inference_steps", 50)
        phenotypes = task.get("phenotypes", "0.5,0.5,0.5,0.5")
        
        logging.info(f"Generating {len(seeds)} videos for prompt: '{prompt}'")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Phenotype values: {phenotypes}")
        
        # Parse phenotype string to list of floats
        if isinstance(phenotypes, str):
            phenotype_values = [float(x) for x in phenotypes.split(",")]
        
        # Convert to tensor
        phenotypes_tensor = torch.tensor([phenotype_values], dtype=torch.float16 if dtype == torch.float16 else torch.float)
        logging.info(f"Will inject phenotypes: {phenotypes_tensor}")
        
        # Generate videos for all seeds
        for seed in seeds:
            output_path = os.path.join(output_dir, f"seed{seed}.mp4")
            logging.info(f"Generating video with seed {seed}, saving to {output_path}")
            
            # Set up generator for reproducibility
            generator = torch.Generator().manual_seed(seed)
            
            # Generate the video
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                phenotypes=phenotypes_tensor,  # Pass phenotypes directly to the pipeline
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Save the video
            frames = result.frames[0]
            export_to_video(frames, output_path, fps=fps)
            logging.info(f"Saved video to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch generate phenotype-controlled videos using CogVideoX")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file")
    args = parser.parse_args()
    
    generate_videos_from_config(args.config)

if __name__ == "__main__":
    main() 