#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Accelerate for device management
from accelerate import Accelerator
# Diffusers: pipeline + scheduler
from diffusers import StableDiffusionPipeline, DDPMScheduler

###############################################################################
#                           USER-DEFINED VARIABLES                            #
###############################################################################
MODEL_PATH = "models/sd2-mca-frame-embeddings/final"       # Folder with unet/, vae/, etc.
FRAME_EMBED_PATH = "models/sd2-mca-frame-embeddings/final/frame_embed.pt"
SAVE_PATH = "./generated_frames.png"                         # Where to save the final PNG

MAX_FRAME_IDX = 40         # Must match how many embeddings you trained
HIDDEN_DIM = 1024          # Typically 768 or 1024 for SD2

FRAMES = [0, 9, 19, 29, 39]   # Which frames to generate (1 row each)
SAMPLES_PER_FRAME = 6         # How many random samples per frame (columns)
NUM_INFERENCE_STEPS = 25      # Fewer steps = faster sampling (lower quality)
###############################################################################


def merge_images_side_by_side(rows_of_images):
    """
    Given a list of rows, where each row is a list of PIL images,
    create a single large PIL image with each row and column arranged.
    """
    # Compute row heights and row widths
    row_heights = []
    row_widths = []
    for row_imgs in rows_of_images:
        heights = [img.height for img in row_imgs]
        widths = [img.width for img in row_imgs]
        row_heights.append(max(heights))
        row_widths.append(sum(widths))

    final_width = max(row_widths)
    final_height = sum(row_heights)

    merged = Image.new("RGB", (final_width, final_height), (255, 255, 255))
    y_offset = 0

    for row_idx, row_imgs in enumerate(rows_of_images):
        x_offset = 0
        for img in row_imgs:
            merged.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_heights[row_idx]

    return merged


def generate_grid_of_frames(
    unet_tmp,
    vae_tmp,
    frame_embed_tmp,
    scheduler,
    frames,
    samples_per_frame,
    num_inference_steps,
    device
):
    """
    Generates a grid of images for each frame in `frames`:
      - 1 row per frame
      - N columns = samples_per_frame (random seeds)
    Returns a PIL image of the entire grid.
    """
    unet_tmp.eval()
    vae_tmp.eval()
    frame_embed_tmp.eval()

    rows_of_images = []

    with torch.no_grad():
        for frame_idx in frames:
            # We'll generate a single row of images for this frame
            row_images = []

            # Prepare random latents
            batch_size = samples_per_frame
            latents = torch.randn(
                (batch_size, unet_tmp.in_channels, 64, 64),
                device=device
            )

            # Frame embedding: shape (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
            frame_tensor = torch.tensor([frame_idx]*batch_size, device=device, dtype=torch.long)
            cond_emb = frame_embed_tmp(frame_tensor).unsqueeze(1)

            # Re-init the scheduler for the chosen number of steps
            scheduler.set_timesteps(num_inference_steps, device=device)

            # Diffusion sampling loop
            for t in scheduler.timesteps:
                # Only autocast if device is CUDA
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    model_out = unet_tmp(latents, t, encoder_hidden_states=cond_emb).sample
                latents = scheduler.step(model_out, t, latents).prev_sample

            # Decode all latents
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                decoded = vae_tmp.decode(1 / 0.18215 * latents).sample
            decoded = (decoded / 2 + 0.5).clamp(0,1)  # scale from [-1,1] to [0,1]

            # Convert to PIL
            decoded_np = decoded.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
            for i in range(batch_size):
                img_arr = (decoded_np[i] * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_arr)
                row_images.append(pil_img)

            rows_of_images.append(row_images)

    # Merge all rows into one big image
    grid_image = merge_images_side_by_side(rows_of_images)
    return grid_image


def main():
    # 1) Prepare the device via Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # 2) Load the pipeline from disk
    print(f"Loading model from: {MODEL_PATH}")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH)

    # Overwrite the scheduler with a straightforward DDPMScheduler or keep the pipeline's default
    print("Creating a DDPMScheduler for sampling...")
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    unet_tmp = pipe.unet
    vae_tmp = pipe.vae

    # 3) Load frame embedding
    print(f"Loading frame embeddings from: {FRAME_EMBED_PATH}")
    frame_embed_tmp = nn.Embedding(MAX_FRAME_IDX, HIDDEN_DIM)
    frame_embed_tmp.load_state_dict(torch.load(FRAME_EMBED_PATH, map_location="cpu"))

    # 4) Move everything to device
    unet_tmp, vae_tmp, frame_embed_tmp = accelerator.prepare(
        unet_tmp, vae_tmp, frame_embed_tmp
    )
    unet_tmp.to(device)
    vae_tmp.to(device)
    frame_embed_tmp.to(device)

    # 5) Generate the grid
    print("Generating images...")
    grid_image = generate_grid_of_frames(
        unet_tmp=unet_tmp,
        vae_tmp=vae_tmp,
        frame_embed_tmp=frame_embed_tmp,
        scheduler=pipe.scheduler,
        frames=FRAMES,
        samples_per_frame=SAMPLES_PER_FRAME,
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=device
    )

    # 6) Save the output
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    grid_image.save(SAVE_PATH)
    print(f"Saved grid to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
