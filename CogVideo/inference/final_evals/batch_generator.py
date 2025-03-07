"""
Batch generator for CogVideoX evaluations.
This script loads the model once and generates multiple videos with different seeds.
"""

import argparse
import logging
import os
import json
import torch
import sys
from typing import List, Optional, Dict, Any

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel
)
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO)

def setup_pipe(
    model_path: str,
    sft_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """Set up the CogVideoX pipeline with the specified model."""
    logging.info(f"Loading model from {model_path}")
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    
    if sft_path:
        logging.info(f"Loading SFT transformer from {sft_path}")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            sft_path,
            subfolder="",
            torch_dtype=dtype
        )
        pipe.transformer = transformer
        logging.info(f"Successfully replaced transformer with SFT model")
    
    if lora_path:
        logging.info(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1)
    
    # Set up scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    # Enable CPU offload for memory efficiency
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe

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
    
    # Load the model once
    pipe = setup_pipe(model_path, sft_path, lora_path, lora_rank, dtype)
    
    # Process each generation task
    for task in config.get("tasks", []):
        prompt = task.get("prompt", "Time-lapse microscopy video of cells.")
        output_dir = task.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        seeds = task.get("seeds", [42])
        width = task.get("width", None)
        height = task.get("height", None)
        num_frames = task.get("num_frames", 81)
        fps = task.get("fps", 10)
        guidance_scale = task.get("guidance_scale", 8.0)
        num_inference_steps = task.get("num_inference_steps", 50)
        phenotypes = task.get("phenotypes", None)
        
        logging.info(f"Generating {len(seeds)} videos for prompt: '{prompt}'")
        logging.info(f"Output directory: {output_dir}")
        if phenotypes:
            logging.info(f"Phenotype values: {phenotypes}")
        
        # Generate videos for all seeds
        for seed in seeds:
            output_path = os.path.join(output_dir, f"seed{seed}.mp4")
            logging.info(f"Generating video with seed {seed}, saving to {output_path}")
            
            # Generate the video
            if phenotypes:
                # Parse phenotypes string to list of floats if it's a string
                if isinstance(phenotypes, str):
                    phenotype_values = [float(x) for x in phenotypes.split(",")]
                else:
                    phenotype_values = phenotypes
                
                # Access the CLI demo2's specific phenotype-enabled forward method
                kwargs = {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_frames": num_frames,
                    "use_dynamic_cfg": True,
                    "generator": torch.Generator().manual_seed(seed),
                }
                
                # Set phenotype conditioning with the pipe
                phenotype_tensor = torch.tensor([phenotype_values], dtype=torch.float16 if dtype == torch.float16 else torch.float)
                if hasattr(pipe, "set_phenotype_tensor"):
                    pipe.set_phenotype_tensor(phenotype_tensor)
                else:
                    logging.warning("Pipeline doesn't have 'set_phenotype_tensor' method. Using standard generation.")
                
                video_frames = pipe(**kwargs).frames[0]
            else:
                # Standard text-to-video generation
                video_frames = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),
                ).frames[0]
            
            # Save the video
            export_to_video(video_frames, output_path, fps=fps)
            logging.info(f"Saved video to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch generate videos using CogVideoX")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file")
    args = parser.parse_args()
    
    generate_videos_from_config(args.config)

if __name__ == "__main__":
    main() 