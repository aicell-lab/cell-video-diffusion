"""
This script demonstrates how to generate a video using a fully finetuned (SFT) CogVideoX model.
It loads a base model architecture and then applies weights from a converted DeepSpeed ZeRO checkpoint.

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo_sft.py --prompt "A girl riding a bike." --base_model_path "THUDM/CogVideoX1.5-5b-I2V" --weights_path "path/to/fp32_model" --generate_type "i2v" --image_path "path/to/image.jpg"
```
"""

import argparse
import logging
import os
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video


logging.basicConfig(level=logging.INFO)

def generate_video(
    prompt: str,
    base_model_path: str,
    weights_path: str,
    num_frames: int = 81,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a video using a fully finetuned (SFT) model.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - base_model_path (str): The path to the base model architecture.
    - weights_path (str): The path to the converted SFT model weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process.
    - num_frames (int): Number of frames to generate.
    - guidance_scale (float): The scale for classifier-free guidance.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1. Load the base model architecture
    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(base_model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(base_model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(base_model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # 2. Load the SFT weights
    logging.info(f"Loading SFT weights from {weights_path}")
    
    # Check if the weights are sharded
    index_file = os.path.join(weights_path, "pytorch_model.bin.index.json")
    if os.path.exists(index_file):
        # Load sharded weights
        import json
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        state_dict = {}
        
        for param_name, filename in weight_map.items():
            shard_file = os.path.join(weights_path, filename)
            shard = torch.load(shard_file, map_location="cpu")
            if param_name in shard:
                state_dict[param_name] = shard[param_name]
    else:
        # Load single file weights
        weights_file = os.path.join(weights_path, "pytorch_model.bin")
        state_dict = torch.load(weights_file, map_location="cpu")
    
    # Load the weights into the model
    pipe.load_state_dict(state_dict, strict=False)

    # 3. Set Scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 4. Enable CPU offload for the model
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 5. Generate the video frames based on the prompt
    if generate_type == "i2v":
        video_generate = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using SFT CogVideoX model")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image or video to be used as input",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX1.5-5B-I2V", help="Path to the base model architecture"
    )
    parser.add_argument(
        "--weights_path", type=str, required=True, help="Path to the converted SFT model weights"
    )
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path to save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        base_model_path=args.base_model_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    ) 