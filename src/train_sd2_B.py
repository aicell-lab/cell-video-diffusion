# File: src/train_sd2_B.py

import math
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from accelerate import Accelerator
import wandb

# Import the SINGLE-CHANNEL dataset class for Approach B
from data.mca_dataset import MCAFrameDatasetSingleChannel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--train_data_dir", type=str, default="data/processed/mca_frame0_256/train")
    parser.add_argument("--output_dir", type=str, default="./sd2-mca-finetuned-B")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    # WandB config
    parser.add_argument("--wandb_project", type=str, default="sd2_cell_finetune_B")
    parser.add_argument("--wandb_run_name", type=str, default="approachB_fullfinetune")
    parser.add_argument("--log_frequency", type=int, default=100, help="Log every X steps.")
    parser.add_argument("--sample_prompts", type=str, nargs="+", default=["a microscopy image of a cell"])
    return parser.parse_args()

def collate_fn(examples):
    """
    Batching function. We expect each example to have:
      - pixel_values: shape [1, 256, 256]
      - text: a string prompt
    """
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])  # [batch, 1, 256, 256]
    texts = [ex["text"] for ex in examples]
    return {
        "pixel_values": pixel_values,
        "text": texts
    }

def adapt_vae_for_single_channel(vae):
    """
    Modifies the VAE so that vae.encoder.conv_in has in_channels=1
    rather than 3. We do this by:
      1) Creating a new Conv2d layer with in_channels=1, same out_channels, kernel_size, etc.
      2) Initializing its weights by averaging over the old 3 channels.
      3) Replacing the old conv_in layer in the encoder.
    """
    old_conv = vae.encoder.conv_in
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    dilation = old_conv.dilation
    use_bias = (old_conv.bias is not None)

    # Create new conv with 1 input channel
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=use_bias
    )

    # Average old weights across the 3 input channels
    with torch.no_grad():
        # old_conv.weight shape = [out_channels, 3, kH, kW]
        old_weight = old_conv.weight.data
        # shape => [out_channels, 1, kH, kW]
        new_weight = old_weight.mean(dim=1, keepdim=True)

        # Assign new weights
        new_conv.weight.data = new_weight.clone()
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    # Replace the layer
    vae.encoder.conv_in = new_conv

def main():
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # ==================================================
    # 1) Load the Pretrained SD2 Model
    # ==================================================
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)

    # Overwrite default pipeline scheduler with DDPMScheduler for training
    pipe.scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Extract model components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # ==================================================
    # 2) Adapt the VAE for Single-Channel Input
    # ==================================================
    adapt_vae_for_single_channel(vae)

    # ==================================================
    # 3) Freeze or Unfreeze Modules
    # ==================================================
    # By default, let's freeze the VAE & text encoder, train only U-Net:
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    # ==================================================
    # 4) Create Dataset & Dataloaders
    # ==================================================
    train_transform = transforms.Compose([
        transforms.ToTensor(),               # -> shape [1, H, W]
        transforms.Normalize([0.5], [0.5])   # single channel => one mean/std
    ])

    train_dataset = MCAFrameDatasetSingleChannel(
        image_dir=args.train_data_dir,
        prompt="a microscopy image of a cell",
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

    # ==================================================
    # 5) Setup Optimizer & LR Scheduler
    # ==================================================
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    # If you want to fine-tune the VAE as well, uncomment:
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

    # Prepare everything with Accelerator
    unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # --------------------------------------------------
    # Helper function to generate sample images & log
    # --------------------------------------------------
    def log_generated_images(step):
        # Create a pipeline with the updated weights
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

        wandb.log({
            "generated_samples": [
                wandb.Image(img, caption=cap)
                for img, cap in zip(images, captions)
            ],
            "global_step": step
        })

    # ==================================================
    # 6) Training Loop
    # ==================================================
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1) Tokenize text
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
                text_embeddings = text_encoder(**text_inputs).last_hidden_state

                # 2) Encode images to latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # SD scaling factor

                # 3) Sample random timesteps & add noise
                noise_scheduler = pipe.scheduler
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()

                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4) U-Net forward pass
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=text_embeddings
                ).sample

                # 5) Target is the noise
                loss = F.mse_loss(model_pred, noise)

                # 6) Backprop
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process and step % args.log_frequency == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "step": global_step})

            global_step += 1

        # End of epoch: log sample images
        if accelerator.is_main_process:
            log_generated_images(global_step)

    # ==================================================
    # 7) Save Final Model
    # ==================================================
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
        print(f"Model saved to {final_dir}")

        wandb.finish()

if __name__ == "__main__":
    main()
