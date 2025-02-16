import math
import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from accelerate import Accelerator

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.optimization import get_scheduler

from data.mca_dataset import MCAFrameDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--train_data_dir", type=str, default="data/processed/mca_frame0_256/train")
    parser.add_argument("--output_dir", type=str, default="./sd2-mca-finetuned-frames")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Additional
    parser.add_argument("--max_frame_idx", type=int, default=40, help="Total frames (so embedding has shape [40, hidden_dim])")

    # WandB logging
    parser.add_argument("--wandb_project", type=str, default="sd2_cell_finetune")
    parser.add_argument("--wandb_run_name", type=str, default="frame_embedding_approach")
    parser.add_argument("--log_frequency", type=int, default=100)
    parser.add_argument("--sample_frames", type=int, nargs="+", default=[0, 10, 20, 30, 39])
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # 1) Load the base SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # We only need the UNet and VAE from the pipeline. 
    unet = pipe.unet
    vae = pipe.vae

    # *Optionally* disable gradient for the text encoder since we won't use it
    if pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)

    vae.requires_grad_(False)
    unet.requires_grad_(True)

    # 2) Figure out the latent dimension used by UNet cross-attention
    #    stable-diffusion-2-base typically uses text_encoder hidden_dim=1024 or 768.
    #    We'll read from the text_encoder if present, else assume 1024 or 768 manually.
    if pipe.text_encoder is not None and hasattr(pipe.text_encoder.config, "hidden_size"):
        hidden_dim = pipe.text_encoder.config.hidden_size
    else:
        hidden_dim = 1024  # or 768, depending on your model variant

    print(f"UNet hidden dim: {hidden_dim}")

    # 3) Define a trainable Embedding for frame indices
    #    shape: (num_frames, hidden_dim)
    frame_embed = nn.Embedding(args.max_frame_idx, hidden_dim)

    # 4) Create the dataset & dataloader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = MCAFrameDataset(
        image_dir=args.train_data_dir,
        transform=train_transform
    )
    info = train_dataset.get_dataset_info()
    print(f"Dataset size: {info['size']} images")
    print(f"Batch size: {args.train_batch_size}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 5) Setup optimizer, scheduler
    #    We train UNet + the frame_embed table
    params_to_optimize = list(unet.parameters()) + list(frame_embed.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps
    )

    # 6) Prepare everything with Accelerator
    unet, vae, frame_embed, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, frame_embed, optimizer, train_dataloader, lr_scheduler
    )


    def log_generated_images(step):
        """
        Generates images for each frame in args.sample_frames **in one batch**.
        This is much faster than doing a full diffusion loop per frame.

        We'll do fewer inference steps (e.g. 25) for speed.
        """
        unet_tmp = accelerator.unwrap_model(unet)
        vae_tmp = accelerator.unwrap_model(vae)
        frame_embed_tmp = accelerator.unwrap_model(frame_embed)

        frames = args.sample_frames  # e.g. [0, 9, 19, 29, 39]
        batch_size = len(frames)

        # Initialize random latents for the entire batch
        latents = torch.randn(
            (batch_size, unet_tmp.in_channels, 64, 64),
            device=accelerator.device
        )

        # Retrieve frame embeddings: shape (batch_size, hidden_dim)
        frame_idxs = torch.tensor(frames, device=latents.device, dtype=torch.long)
        cond_emb = frame_embed_tmp(frame_idxs)
        cond_emb = cond_emb.unsqueeze(1)  # becomes (batch_size, 1, hidden_dim)

        # Use fewer sampling steps for quick previews
        num_inference_steps = 25
        pipe.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop (batched)
        unet_tmp.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            for t in pipe.scheduler.timesteps:
                # (batch_size, in_channels, 64, 64) forward pass
                model_out = unet_tmp(latents, t, encoder_hidden_states=cond_emb).sample
                latents = pipe.scheduler.step(model_out, t, latents).prev_sample

            # Decode the entire batch
            decoded = vae_tmp.decode(1 / 0.18215 * latents).sample  # (batch_size, 3, 512, 512) typically
            # Rescale from [-1,1] to [0,1]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

        # Convert each sample to PIL and store in a list
        images_out = []
        decoded_np = decoded.permute(0, 2, 3, 1).cpu().numpy()  # (batch_size, height, width, 3)
        for i in range(batch_size):
            img = (decoded_np[i] * 255).astype("uint8")
            image_pil = Image.fromarray(img)
            images_out.append(image_pil)

        # Merge images side-by-side
        merged = merge_images_side_by_side(images_out)
        wandb.log({
            "global_step": step,
            "sampled_frames": wandb.Image(merged, caption=f"Frames: {frames}")
        })


    def merge_images_side_by_side(images):
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        merged = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
        x_offset = 0
        for img in images:
            merged.paste(img, (x_offset, 0))
            x_offset += img.width
        return merged

    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        frame_embed.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                frame_idxs = batch["frame_idx"].to(accelerator.device)

                # 7.1) Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()  # (B,4,64,64)
                latents = latents * 0.18215

                # 7.2) Sample random timesteps
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()

                # 7.3) Add noise
                noise = torch.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # 7.4) Frame embedding => shape (B, 1, hidden_dim)
                frame_embeds = frame_embed(frame_idxs)       # (B, hidden_dim)
                frame_embeds = frame_embeds.unsqueeze(1)    # (B,1,hidden_dim)

                # 7.5) U-Net forward pass, ignoring text
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=frame_embeds).sample

                # 7.6) Loss = MSE with the added noise
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                if step % args.log_frequency == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    wandb.log({"train/loss": loss.item(), "step": global_step})

                # Periodic sampling
                if global_step % 500 == 0:
                    log_generated_images(global_step)

            global_step += 1

    # Save final model
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet_to_save = accelerator.unwrap_model(unet)
        frame_embed_to_save = accelerator.unwrap_model(frame_embed)
        vae_to_save = accelerator.unwrap_model(vae)

        # Save pipeline (Note: stable diffusion pipeline expects text_encoder, 
        # but we can still call `save_pretrained` to store unet, vae, etc.)
        pipe.save_pretrained(
            final_dir,
            unet=unet_to_save,
            vae=vae_to_save,
            text_encoder=None  # or just leave as None
        )
        # Save frame embedding separately
        torch.save(frame_embed_to_save.state_dict(), os.path.join(final_dir, "frame_embed.pt"))

        print("Fine-tuning complete! Model + embeddings saved at:", final_dir)
        wandb.finish()

if __name__ == "__main__":
    main()
