"""
Batch generator for CogVideoX evaluations.
This script loads the model once and generates multiple videos with different seeds.
Based on cli_demo.py functionality (standard text-to-video).
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
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXTransformer3DModel
)
from diffusers.utils import export_to_video, load_image, load_video

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

def setup_pipe(
    model_path: str,
    generate_type: str = "t2v",
    image_or_video_path: Optional[str] = None,
    sft_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """Set up the CogVideoX pipeline with the specified model."""
    logging.info(f"Loading model from {model_path}")
    
    # Determine model resolution
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, RESOLUTION_MAP["fp32_model"])
    
    # Initialize appropriate pipeline based on generation type
    image = None
    video = None
    
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if image_or_video_path:
            image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:  # v2v
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if image_or_video_path:
            video = load_video(image_or_video_path)
    
    # Load SFT model if specified
    if sft_path:
        logging.info(f"Loading SFT transformer from {sft_path}")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            sft_path,
            subfolder="",
            torch_dtype=dtype
        )
        pipe.transformer = transformer
        logging.info(f"Successfully replaced transformer with SFT model")
    
    # Load LoRA weights if specified
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
    
    return pipe, image, video, desired_resolution

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
    image_or_video_path = config.get("image_or_video_path", None)
    
    # Load the model once
    pipe, image, video, default_resolution = setup_pipe(
        model_path, generate_type, image_or_video_path, sft_path, lora_path, lora_rank, dtype
    )
    
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
        
        logging.info(f"Generating {len(seeds)} videos for prompt: '{prompt}'")
        logging.info(f"Output directory: {output_dir}")
        
        # Generate videos for all seeds
        for seed in seeds:
            output_path = os.path.join(output_dir, f"seed{seed}.mp4")
            logging.info(f"Generating video with seed {seed}, saving to {output_path}")
            
            # Generate based on generation type
            if generate_type == "i2v":
                video_frames = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    image=image,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),
                ).frames[0]
            elif generate_type == "t2v":
                video_frames = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),
                ).frames[0]
            else:  # v2v
                video_frames = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    video=video,
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