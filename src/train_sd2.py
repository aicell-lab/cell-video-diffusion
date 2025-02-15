import math
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import wandb
from data.mca_dataset import MCAFrameDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--train_data_dir", type=str, default="data/processed/mca_frame0_256/train")
    parser.add_argument("--output_dir", type=str, default="./sd2-mca-finetuned")
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
    parser.add_argument("--sample_prompts", type=str, nargs="+", default=["a microscopy image of a cell"])
    return parser.parse_args()

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

    # Initialize wandb (only in the main process)
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # 1) Load Pretrained Model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,  # half precision if enough VRAM
        # variant="fp16",
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Extract model components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # 2) Freeze the VAE by default (common approach)
    # Usually, the VAE is either kept frozen or trained lightly. 
    # Freezing saves memory and reduces overfitting risk on small data.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # But we do want to train the U-Net
    unet.requires_grad_(True)

    # 3) Create Dataset & Dataloaders
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # SD expects images in [-1,1]
    ])

    train_dataset = MCAFrameDataset(
        image_dir=args.train_data_dir,
        prompt="a microscopy image of a cell",  # or something short
        transform=train_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 4) Optimizer & Scheduler
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    # If you do want to train VAE as well, uncomment:
    # vae.requires_grad_(True)
    # params_to_optimize += list(vae.parameters())

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps
    )

    # Prepare with accelerator
    unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # A small utility to generate sample images & log them to wandb
    def log_generated_images(step):
        # Create a temporary pipeline with updated weights
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

        images = []
        captions = []

        for prompt in args.sample_prompts:
            with torch.autocast("cuda"):
                out = pipe_tmp(prompt, num_inference_steps=25, guidance_scale=7.5)
            generated_img = out.images[0]
            images.append(generated_img)
            captions.append(prompt)

        # Log to wandb
        wandb.log({
            "generated_samples": [
                wandb.Image(img, caption=cap)
                for img, cap in zip(images, captions)
            ],
            "global_step": step
        })

    # 5) Training Loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train(); text_encoder.train()
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
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
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
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=text_embeddings
                ).sample

                # 5.5) The target is the noise (ε)
                target = noise

                # 5.6) Compute loss & backprop
                loss = torch.nn.functional.mse_loss(model_pred, target)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 5.7) Logging
            if accelerator.is_main_process:
                if step % args.log_frequency == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    wandb.log({"train/loss": loss.item(), "step": global_step})

                # Optionally generate images every few steps
                if step % (args.log_frequency * 10) == 0 and step > 0:
                    log_generated_images(global_step)

            global_step += 1

        # End of epoch checkpoint
        if accelerator.is_main_process:
            pipeline_save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            os.makedirs(pipeline_save_dir, exist_ok=True)
            # Unwrap
            unet_to_save = accelerator.unwrap_model(unet)
            text_encoder_to_save = accelerator.unwrap_model(text_encoder)
            vae_to_save = accelerator.unwrap_model(vae)

            pipe.save_pretrained(
                pipeline_save_dir,
                unet=unet_to_save,
                text_encoder=text_encoder_to_save,
                vae=vae_to_save
            )

    # Final save
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

    print("Fine-tuning complete!")
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()