from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register


class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])

        return ret

    @override
    def compute_loss(self, batch) -> dict:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"] # conditioning images

        # Shape of prompt_embedding: [B, seq_len, hidden_size] = [2, 226, 4096]
        # Shape of latent: [B, C, F, H, W] = [2, 16, 21, 96, 170]
        # Shape of images: [B, C, H, W] = [2, 3, 768, 1360]

        # pad the latent (prepend frames) so that the math works out
        patch_size_t = self.state.transformer_config.patch_size_t # 2
        ncopy = 0
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2) # [2, 16, 22, 96, 170]
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings (this step is just ensuring consistency, shape is unchanged)
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images (referred to in appendix D in the paper)
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype) # convert from log space to linear space
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]

        # encode the noisy images, go from [2, 3, 1, 768, 1360] to [2, 16, 1, 96, 170]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor # [2, 16, 1, 96, 170]

        # Sample a random timestep for each sample (between 0 and 1000)
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4) # [2, 22, 16, 96, 170]
        image_latents = image_latents.permute(0, 2, 1, 3, 4) # [2, 1, 16, 96, 170]
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:]) # [2, 21, 16, 96, 170]
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1) # [2, 22, 16, 96, 170]

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2) # [2, 22, 32, 96, 170]


        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = ( # ([4880, 64], [4880, 64]) for sin/cos embeddings
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        # Predict noise, For CogVideoX1.5 Only.
        # a special feature in CogVideoX 1.5 that helps control the strength of motion/optical flow in the generated video
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer( # [2, 22, 16, 96, 170]
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        # important: we use latent_noisy (which has C=16) here, not latent_img_noisy (which has C=32)!
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        orig_loss = loss.mean()

        if self.args.loss_function == "ffe":
            return self.ffe_loss(orig_loss, latent_pred, latent, ncopy)
        elif self.args.loss_function == "mfe":
            return self.mfe_loss(orig_loss, latent_pred, latent, ncopy)
        
        return {"loss": orig_loss, "components": {}}
    
    def ffe_loss(self, orig_loss, latent_pred, latent, ncopy):
        # Get the actual first frame (skip duplicated frames)
        first_frame_pred = latent_pred[:, ncopy]  # First actual predicted frame
        first_frame_gt = latent[:, ncopy]         # First actual ground truth frame
        
        # Calculate MSE specifically for the first frame
        ffe_loss = torch.mean((first_frame_pred - first_frame_gt) ** 2, dim=(1, 2, 3)).mean()
        ffe_weight = self.args.ffe_weight
        
        total_loss = orig_loss + ffe_weight * ffe_loss
        
        return {
            "loss": total_loss,
            "components": {
                "orig_loss": orig_loss.detach().item(),
                "ffe_loss": ffe_loss.detach().item()
            }
        }

    def mfe_loss(self, orig_loss, latent_pred, latent, ncopy):
        # Number of frames to apply emphasis to
        n_frames = self.args.mfe_num_frames  # e.g., 3
        base_weight = self.args.mfe_weight  # e.g., 5.0
        decay_factor = self.args.mfe_decay  # e.g., 0.5
        
        # Calculate weights for each frame: [base_weight, base_weight*exp(-decay), base_weight*exp(-2*decay), ...]
        frame_indices = torch.arange(n_frames, device=latent_pred.device)
        frame_weights = base_weight * torch.exp(-decay_factor * frame_indices)
        
        # Initialize the total emphasis loss
        mfe_loss = 0.0
        
        # Calculate weighted MSE for each of the first n frames, skipping duplicated frames
        for i in range(n_frames):
            # Adjust index to skip duplicated frames
            actual_idx = i + ncopy
            
            if actual_idx < latent_pred.shape[1]:  # Ensure we don't exceed the number of frames
                frame_pred = latent_pred[:, actual_idx]
                frame_gt = latent[:, actual_idx]
                frame_loss = torch.mean((frame_pred - frame_gt) ** 2, dim=(1, 2, 3)).mean()
                # Use the original i for the weight index (not actual_idx)
                mfe_loss += frame_weights[i] * frame_loss
        
        # Combine with original loss
        total_loss = orig_loss + mfe_loss
        
        return {
            "loss": total_loss,
            "components": {
                "orig_loss": orig_loss.detach().item(),
                "mfe_loss": mfe_loss.detach().item()
            }
        }

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            generator=self.state.generator,
            # num_inference_steps=1, # change this to speedup validation for debugging
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
