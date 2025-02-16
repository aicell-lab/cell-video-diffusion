"""
This script is used to train the SD2 model on the MCA dataset. Each sample is a frame from a microscopy video.
"""
import math
import os
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import wandb
from data.mca_dataset import MCAFrameDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--train_data_dir", type=str, default="data/processed/mca_frame0_256/train")
    parser.add_argument("--output_dir", type=str, default="./sd2-mca-finetuned-A")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    # WandB config
    parser.add_argument("--wandb_project", type=str, default="sd2_cell_finetune")
    parser.add_argument("--wandb_run_name", type=str, default="approachA_fullfinetune")
    parser.add_argument("--log_frequency", type=int, default=100, help="Log every X steps.")
    parser.add_argument("--sample_prompts", type=str, nargs="+",
                        default=["a microscopy image of a cell"])  # multiple prompts
    return parser.parse_args()


def merge_images_side_by_side(images):
    """
    Takes a list of PIL Images and merges them horizontally (side-by-side)
    into a single PIL Image.
    """
    widths, heights = zip(*(img.size for img in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create a new blank image (white background)
    merged = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    
    x_offset = 0
    for img in images:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return merged


def collate_fn(examples):
    """
    after collate_fn, the batch will be:
    {
        "pixel_values": tensor[4, 3, 256, 256],  # Batch of 4 images
        "text": ["a microscopy image of a cell", "a microscopy image of a cell", ...]  # List of 4 prompts
    }
    """
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    texts = [ex["text"] for ex in examples]
    return {
        "pixel_values": pixel_values,
        "text": texts
    }

def main():
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Freeze the VAE and encoder by default, but not the U-Net
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # SD expects images in [-1,1]
    ])

    train_dataset = MCAFrameDataset(
        image_dir=args.train_data_dir,
        base_prompt="a microscopy image of a cell",
        transform=train_transform
    )
    info = train_dataset.get_dataset_info()
    print(f"Dataset size: {info['size']} images")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Each epoch has {math.ceil(len(train_dataset) / args.train_batch_size)} batches")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps
    )

    unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    def log_generated_images(step):
        """
        Generates images for each prompt in args.sample_prompts,
        merges them side-by-side, and logs as one single image to W&B.
        """
        unet_tmp = accelerator.unwrap_model(unet)
        text_encoder_tmp = accelerator.unwrap_model(text_encoder)
        vae_tmp = accelerator.unwrap_model(vae)

        pipe_tmp = StableDiffusionPipeline(
            text_encoder=text_encoder_tmp,
            vae=vae_tmp,
            unet=unet_tmp,
            tokenizer=tokenizer,
            scheduler=pipe.scheduler,
            safety_checker=pipe.safety_checker,
            feature_extractor=pipe.feature_extractor
        ).to(accelerator.device)

        # Generate images for each prompt
        gen_images = []
        for prompt in args.sample_prompts:
            with torch.autocast("cuda"):
                out = pipe_tmp(prompt, num_inference_steps=25, guidance_scale=7.5)
            gen_images.append(out.images[0])

        # Merge them into a single row
        row_image = merge_images_side_by_side(gen_images)

        # Caption can list all prompts for reference
        caption_text = " | ".join(args.sample_prompts)

        # Log to W&B: single image with multiple sub-images side-by-side
        wandb.log({
            "global_step": step,
            "merged_samples": wandb.Image(row_image, caption=caption_text)
        })


    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 5.1) Encode text
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
                text_embeddings = text_encoder(**text_inputs).last_hidden_state

                # 5.2) Encode images to latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # SD scaling

                # 5.3) Random timesteps, add noise
                noise_scheduler = pipe.scheduler
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 5.4) U-Net forward
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

                # 5.5) The target is the noise
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                # 5.6) Backprop & update
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                if step % args.log_frequency == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    wandb.log({"train/loss": loss.item(), "step": global_step})
                
                if global_step % 100 == 0:
                    log_generated_images(global_step)

            global_step += 1

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet_to_save = accelerator.unwrap_model(unet)
        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
        vae_to_save = accelerator.unwrap_model(vae)
        pipe.save_pretrained(
            final_dir,
            unet=unet_to_save,
            text_encoder=text_encoder_to_save,
            vae=vae_to_save
        )
        print("Fine-tuning complete! Model saved at:", final_dir)
        wandb.finish()

if __name__ == "__main__":
    main()