import argparse
import logging
from typing import Literal, Optional
import torch
import os
import json

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import export_to_video, load_image, load_video

###############################################################################
# RECOMMENDED RESOLUTION MAP, ETC. (unchanged)
###############################################################################

RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
    "fp32_model": (768, 1360),
}

logging.basicConfig(level=logging.INFO)

from safetensors import safe_open

import sys
sys.path.append("..")
from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder
from finetune.models.modules.combined_model import CombinedTransformerWithEmbedder


def load_sft_combined_model(
    base_dir: str,
    device: torch.device = torch.device("cuda"),
    phenotype_module: str = "single",
    phenotype_dim: int = 4,
    phenotype_hidden_dim: int = 256,
    phenotype_output_dim: int = 4096,
    phenotype_dropout: float = 0.1,
    strict_load: bool = False,
) -> CombinedTransformerWithEmbedder:
    """
    Loads a CogVideoXTransformer3DModel + PhenotypeEmbedder from FP32 safetensors shards
    in the given base_dir, merges them into a single state_dict, and returns a
    CombinedTransformerWithEmbedder. Moves the model to `device`.

    Args:
        base_dir (str): Path to the directory that contains:
            - `config.json`
            - `diffusion_pytorch_model.safetensors.index.json`
            - `model-00001-of-00005.safetensors`
            - ...
            - `model-00005-of-00005.safetensors`
        device (`torch.device`, optional): Device on which to place the model. Defaults to CUDA if available.
        phenotype_module (str): "single" or "multi", per your training setup.
        phenotype_dim (int): Input dimension of the phenotype data (default 4).
        phenotype_hidden_dim (int): Hidden dimension in the phenotype MLP.
        phenotype_output_dim (int): Output dimension in the phenotype MLP (should match text_embed_dim=4096).
        phenotype_dropout (float): Dropout probability in the phenotype MLP.
        strict_load (bool): Whether to enforce strict matching of state_dict keys. Defaults to False.

    Returns:
        CombinedTransformerWithEmbedder: The loaded model, placed on the specified device.
    """
    # 1. Load config
    config_file = os.path.join(base_dir, "config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # 2. Instantiate base CogVideoXTransformer3DModel from config
    transformer = CogVideoXTransformer3DModel(**config_dict)
    print(f"Instantiated CogVideoXTransformer3DModel from {config_file}")

    # 3. Instantiate the phenotype embedder
    phenotype_embedder = PhenotypeEmbedder(
        input_dim=phenotype_dim,
        hidden_dim=phenotype_hidden_dim,
        output_dim=phenotype_output_dim,
        dropout=phenotype_dropout,
    )
    print(f"Instantiated PhenotypeEmbedder: {phenotype_embedder}")

    # 4. Wrap them in the combined model
    combined_model = CombinedTransformerWithEmbedder(
        transformer=transformer,
        phenotype_embedder=phenotype_embedder,
        phenotype_module=phenotype_module,
    )
    print(f"Created CombinedTransformerWithEmbedder:\n{combined_model}")

    # 5. Merge safetensor shards into a single state_dict
    print(f"Loading safetensors from {base_dir}")
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
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # 7. Eval + move to device
    combined_model.eval()
    combined_model.to(device)
    print("Model loaded and moved to", device)

    return combined_model



def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    sft_path: str = None,
    phenotypes_str: str = None,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    Now supports passing a small phenotype vector, which is injected into the transformer's encoder hidden states.
    """

    ###############################################################################
    # 1. Determine resolution & create pipeline
    ###############################################################################

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, (768, 1360))

    if width is None or height is None:
        height, width = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    else:
        # Warn if the user sets a resolution that doesn't match recommended
        if (height, width) != desired_resolution and generate_type != "i2v":
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. "
                f"Switching back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    # Based on generate_type, load the appropriate pipeline
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image_or_video_path)
        video = None
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = None
        video = None
    else:  # v2v
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = None
        video = load_video(image_or_video_path)

    if sft_path:
        print(f"Loading SFT combined model from {sft_path}")
        # This returns a CombinedTransformerWithEmbedder
        combined_model = load_sft_combined_model(
            base_dir=sft_path,
            device=torch.device("cuda"),
            phenotype_module="single",  # or "multi", whichever you used
            phenotype_dim=4,           # match your training
            phenotype_hidden_dim=256,
            phenotype_output_dim=4096,
            phenotype_dropout=0.1,
            strict_load=False
        )
        # Now we can directly do `pipe.transformer = combined_model`
        pipe.transformer = combined_model
        print("Successfully replaced pipeline.transformer with CombinedTransformerWithEmbedder from SFT!")


    ###############################################################################
    # 3. (Optional) Convert pipeline.transformer -> CombinedTransformerWithEmbedder
    #    if the user provided phenotypes.
    ###############################################################################

    if phenotypes_str is not None:
        # Parse the phenotypes from a comma-separated string into a [batch_size=1, input_dim=4] tensor
        phen_list = [float(x) for x in phenotypes_str.split(",")]
        phenotypes_tensor = torch.tensor(phen_list).unsqueeze(0)  # shape (1, 4) if you have 4 dims
        print(f"Will inject phenotypes: {phenotypes_tensor}")

        # 3A. Instantiate your embedder and combined model
        from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder
        from finetune.models.modules.combined_model import CombinedTransformerWithEmbedder

        phenotype_embedder = PhenotypeEmbedder(
            input_dim=4,    # must match what you used in training
            hidden_dim=256,
            output_dim=4096,
            dropout=0.1
        )

        # Wrap the pipeline's current transformer
        combined_model = CombinedTransformerWithEmbedder(
            transformer=pipe.transformer,
            phenotype_embedder=phenotype_embedder,
            phenotype_module="single",  # or "multi"
        )


        pipe.transformer = combined_model
        print("Replaced pipeline.transformer with CombinedTransformerWithEmbedder + injected phenotypes.")

    ###############################################################################
    # 4. (Optional) Load LoRA, set up scheduler, offload to CPU/GPU, etc.
    ###############################################################################

    # LoRA
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1)

    # Replace with DPMScheduler, or DDIM, etc.
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # CPU offload
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    ###############################################################################
    # 5. Generate the video!
    ###############################################################################
    generator = torch.Generator().manual_seed(seed)
    if generate_type == "i2v":
        result = pipe(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    elif generate_type == "t2v":
        # import pdb; pdb.set_trace()
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            phenotypes=phenotypes_tensor,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    else:  # v2v
        result = pipe(
            prompt=prompt,
            video=video,
            height=height,
            width=width,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    frames = result.frames[0]
    export_to_video(frames, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_or_video_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--sft_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output.mp4")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--generate_type", type=str, default="t2v")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--phenotypes",
        type=str,
        default=None,
        help="Comma-separated list of phenotype floats, e.g. '0.1,0.5,0.0,0.7'.",
    )

    args = parser.parse_args()

    # Parse dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        sft_path=args.sft_path,
        phenotypes_str=args.phenotypes,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=torch_dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )
