from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.models.modules.phenotype_embedder import PhenotypeEmbedder, PhenotypeEmbedderMulti

from ..utils import register


class CogVideoXT2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        # Initialize phenotype embedder if phenotype conditioning is enabled
        if self.args.use_phenotype_conditioning:
            # Get transformer hidden size to ensure compatibility
            text_hidden_size = components.text_encoder.config.hidden_size if hasattr(components.text_encoder, 'config') else 4096
            
            # Initialize the appropriate phenotype embedder based on configuration
            if self.args.phenotype_module == "single":
                components.phenotype_embedder = PhenotypeEmbedder(
                    input_dim=4,  # Four phenotype features
                    hidden_dim=256,
                    output_dim=text_hidden_size,  # Match text encoder hidden size
                    dropout=0.1
                )
            elif self.args.phenotype_module == "multi":
                components.phenotype_embedder = PhenotypeEmbedderMulti(
                    input_dim=4,  # Four phenotype features
                    hidden_dim=256,
                    output_dim=text_hidden_size,  # Match text encoder hidden size
                    dropout=0.1
                )
            else:
                raise ValueError(f"Unknown phenotype module type: {self.args.phenotype_module}")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
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
        ret = {"encoded_videos": [], "prompt_embedding": []}
        
        # Add phenotype collection when enabled
        if self.args.use_phenotype_conditioning:
            ret["phenotypes"] = []

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            
            # Collect phenotype data when enabled
            if self.args.use_phenotype_conditioning:
                ret["phenotypes"].append(sample["phenotype"])

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        
        # Stack phenotypes when enabled
        if self.args.use_phenotype_conditioning:
            ret["phenotypes"] = torch.stack(ret["phenotypes"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        
        # Process phenotype data if enabled
        if self.args.use_phenotype_conditioning and "phenotypes" in batch:
            import pdb; pdb.set_trace()
            # Get phenotype embeddings using our trainable embedder
            phenotype_data = batch["phenotypes"]
            phenotype_embedding = self.components.phenotype_embedder(phenotype_data)
            
            # Handle combining embeddings based on phenotype module type
            if self.args.phenotype_module == "single":
                # Single token case: prepend one token and discard last token from text
                # phenotype_embedding shape: [batch_size, 1, hidden_size]
                # prompt_embedding shape: [batch_size, seq_len, hidden_size]
                prompt_embedding = torch.cat([phenotype_embedding, prompt_embedding[:, :-1, :]], dim=1)
            else:  # "multi"
                # Multi-token case: prepend 4 tokens and discard last 4 tokens from text
                # phenotype_embedding shape: [batch_size, 4, hidden_size]
                # prompt_embedding shape: [batch_size, seq_len, hidden_size]
                # We need to discard the last 4 tokens to maintain the same sequence length
                tokens_to_discard = phenotype_embedding.size(1)  # Should be 4
                prompt_embedding = torch.cat(
                    [phenotype_embedding, prompt_embedding[:, :-tokens_to_discard, :]], 
                    dim=1
                )

        # Shape of prompt_embedding: [B, seq_len, hidden_size] when phenotype conditioning is not used
        # Shape of prompt_embedding: [B, seq_len+1, hidden_size] when single token phenotype conditioning is used
        # Shape of prompt_embedding: [B, seq_len+4, hidden_size] when multi token phenotype conditioning is used
        
        # Shape of latent: [B, C, F, H, W] = [2, 16, 21, 96, 170]

        # pad the latent (prepend frames) so that the math works out
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2) # [2, 16, 22, 96, 170]
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
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

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return {"loss": loss, "components": {}}

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        prompt = eval_data["prompt"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
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


register("cogvideox-t2v", "lora", CogVideoXT2VLoraTrainer)
