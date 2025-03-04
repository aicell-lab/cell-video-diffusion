import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    load_phenotypes,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor

def count_nuclei_basic_threshold(frame_bgr, threshold=50, min_area=5):
    """
    Simple nucleus count via threshold + connected components.
    frame_bgr: (H, W, 3) BGR image
    threshold: brightness threshold for binarization
    min_area : to ignore small noise
    returns: integer count of distinct 'nuclei'
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    num_labels, labels = cv2.connectedComponents(bin_img)

    # Exclude small or background
    count = 0
    for label_id in range(1, num_labels):
        area = np.sum(labels == label_id)
        if area >= min_area:
            count += 1
    return count

def count_first_last_nuclei(video_path, threshold=50, min_area=5):
    """
    Reads the first and last frame of 'video_path', 
    returns (#nuclei_first, #nuclei_last, ratio).
    """
    cap = cv2.VideoCapture(str(video_path))  # ensure string
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames_count < 2:
        cap.release()
        return None, None, None

    # First frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None, None, None
    count_first = count_nuclei_basic_threshold(first_frame, threshold=threshold, min_area=min_area)

    # Last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_count - 1)
    ret, last_frame = cap.read()
    if not ret:
        cap.release()
        return count_first, None, None
    count_last = count_nuclei_basic_threshold(last_frame, threshold=threshold, min_area=min_area)
    cap.release()

    ratio = 0 if count_first == 0 else count_last / count_first
    return count_first, count_last, ratio

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )
        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        
        if self.args.model_type == "i2v":
            self.dataset = I2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Prepare PhenotypeEmbedder if phenotype conditioning is enabled
        if self.args.use_phenotype_conditioning and self.components.phenotype_embedder is not None:
            self.components.phenotype_embedder.requires_grad_(False)  # Freeze during data preparation, unfreeze in prepare_trainable_parameters
            self.components.phenotype_embedder = self.components.phenotype_embedder.to(
                self.accelerator.device, dtype=self.state.weight_dtype
            )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
        
        # Handle phenotype embedder separately for clarity
        if self.args.use_phenotype_conditioning and hasattr(self.components, "phenotype_embedder") and self.components.phenotype_embedder is not None:
            logger.info("Setting phenotype_embedder to be trainable")
            self.components.phenotype_embedder.requires_grad_(True)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer and phenotype_embedder), 
        # and cast them to the specified data type
        ignore_list = ["transformer"]
        if self.args.use_phenotype_conditioning and hasattr(self.components, "phenotype_embedder") and self.components.phenotype_embedder is not None:
            ignore_list.append("phenotype_embedder")
        ignore_list += self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)
        
        # Also cast phenotype_embedder params if it exists and is enabled
        if self.args.use_phenotype_conditioning and hasattr(self.components, "phenotype_embedder") and self.components.phenotype_embedder is not None:
            cast_training_params([self.components.phenotype_embedder], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        
        # Add phenotype_embedder parameters to optimization if enabled
        phenotype_embedder_parameters = []
        if self.args.use_phenotype_conditioning and hasattr(self.components, "phenotype_embedder") and self.components.phenotype_embedder is not None:
            phenotype_embedder_parameters = list(filter(lambda p: p.requires_grad, self.components.phenotype_embedder.parameters()))
            if phenotype_embedder_parameters:
                phenotype_parameters_with_lr = {
                    "params": phenotype_embedder_parameters,
                    "lr": self.args.learning_rate,  # Using the same learning rate
                }
                params_to_optimize.append(phenotype_parameters_with_lr)
                logger.info(f"Added {sum(p.numel() for p in phenotype_embedder_parameters)} phenotype embedder parameters to optimization")
        
        # Update total trainable parameter count
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters) + sum(p.numel() for p in phenotype_embedder_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        # Prepare phenotype_embedder if it exists and is enabled
        if self.args.use_phenotype_conditioning and hasattr(self.components, "phenotype_embedder") and self.components.phenotype_embedder is not None:
            self.components.transformer, self.components.phenotype_embedder, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
                self.components.transformer, self.components.phenotype_embedder, self.optimizer, self.data_loader, self.lr_scheduler
            )
        else:
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)
        
        # Load phenotype data for validation if enabled
        validation_phenotypes = [None] * len(validation_prompts)
        if self.args.use_phenotype_conditioning and hasattr(self.args, "validation_phenotypes") and self.args.validation_phenotypes is not None:
            try:
                phenotype_path = self.args.validation_dir / self.args.validation_phenotypes
                validation_phenotypes = load_phenotypes(phenotype_path)
                
                # If phenotype file has fewer entries than prompts, pad with None
                if len(validation_phenotypes) < len(validation_prompts):
                    validation_phenotypes = validation_phenotypes + [None] * (len(validation_prompts) - len(validation_phenotypes))
                    logger.warning(f"Phenotype file has fewer entries ({len(validation_phenotypes)}) than prompts ({len(validation_prompts)})")
                
                logger.info(f"Loaded {len(validation_phenotypes)} phenotype entries for validation")
            except Exception as e:
                logger.warning(f"Failed to load validation phenotypes from {self.args.validation_phenotypes}: {e}")
                validation_phenotypes = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos
        self.state.validation_phenotypes = validation_phenotypes

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss_dict = self.compute_loss(batch)
                    loss = loss_dict["loss"]
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]

                # Add component losses to logs
                if "components" in loss_dict:
                    for name, value in loss_dict["components"].items():
                        logs[name] = value

                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, global_step: int) -> None:
        logger.info(f"Starting validation on process {self.accelerator.process_index}")
        
        # Create validation directory at the beginning
        validation_path = self.args.output_dir / "validation_res"
        validation_path.mkdir(parents=True, exist_ok=True)
        
        if self.args.model_type == "i2v":
            self.validate_i2v(global_step)
        else:
            self.validate_t2v(global_step)

    def validate_t2v(self, step: int) -> None:
        logger.info(f"Starting text-to-video validation on process {self.accelerator.process_index}")
        
        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)
        
        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return
        
        self.components.transformer.eval()
        torch.set_grad_enabled(False)
        
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")
        
        # Initialize pipeline
        pipe = self.initialize_pipeline()
        
        # Handle device placement based on training setup
        if self.state.using_deepspeed:
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            pipe.enable_model_cpu_offload(device=self.accelerator.device)
            pipe = pipe.to(dtype=self.state.weight_dtype)
        
        # Process validation samples
        all_processes_artifacts = []
        for i in range(num_validation_samples):
            # Skip samples not assigned to this process
            if i % accelerator.num_processes != accelerator.process_index:
                continue
            
            prompt = self.state.validation_prompts[i]
            logger.info(f"Process {accelerator.process_index}, i: {i}, Prompt: {prompt}", main_process_only=False)
            
            # Run validation step with just the prompt for t2v
            validation_artifacts = self.validation_step({"prompt": prompt}, pipe)
            
            # Process generated artifacts
            for k, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                if artifact_type not in ["video"] or artifact_value is None:
                    continue
                    
                # Generate filename and save video
                prompt_filename = string_to_filename(prompt)[:25]
                hash_suffix = hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]
                gen_filename = f"validation-gen-{step}-{i}-{prompt_filename}-{hash_suffix}.mp4"
                validation_path = self.args.output_dir / "validation_res"
                gen_path = str(validation_path / gen_filename)
                
                logger.info(f"Process {accelerator.process_index}, i: {i}, Saving generated video to {gen_path}", main_process_only=False)
                export_to_video(artifact_value, gen_path, fps=self.args.gen_fps)
                
                # Log to wandb
                video_wandb = wandb.Video(gen_path, caption=f"Sample {i} - {prompt}")
                all_processes_artifacts.append(video_wandb)
        
        # Gather artifacts from all processes
        all_artifacts = gather_object(all_processes_artifacts)
        
        # Log to wandb from main process
        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(
                        {"validation": {"videos": all_artifacts}},
                        step=step,
                    )
        
        # Clean up
        if self.state.using_deepspeed:
            del pipe
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
            cast_training_params([self.components.transformer], dtype=torch.float32)
            
        free_memory()
        accelerator.wait_for_everyone()
        
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        
        torch.set_grad_enabled(True)
        self.components.transformer.train()
    
    def validate_i2v(self, step: int) -> None:
        logger.info(f"Starting validation on process {self.accelerator.process_index}")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################
        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue
            
            # Always distribute samples across processes, regardless of whether using DeepSpeed
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            prompt = self.state.validation_prompts[i]
            logger.info(f"Process {accelerator.process_index}, i: {i}, Prompt: {prompt}", main_process_only=False)
            image = self.state.validation_images[i]
            logger.info(f"Process {accelerator.process_index}, i: {i}, Image: {image}", main_process_only=False)
            video = self.state.validation_videos[i]
            logger.info(f"Process {accelerator.process_index}, i: {i}, Video: {video}", main_process_only=False)

            # Process input image if it exists
            if image is not None:
                image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)
                
                # Save and log conditioning image
                prompt_filename = string_to_filename(prompt)[:25]
                hash_suffix = hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]
                image_filename = f"validation-{step}-{i}-{prompt_filename}-{hash_suffix}.png"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                image_path = str(validation_path / image_filename)
                
                logger.info(f"Process {accelerator.process_index}, i: {i}, Saving conditioning image to {image_path}", main_process_only=False)
                image.save(image_path)
                image_wandb = wandb.Image(image_path, caption=f"Sample {i} - Conditioning frame - {prompt}")
                all_processes_artifacts.append(image_wandb)

            # Process input video if it exists
            real_counts = None
            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
                
                # Save real video (but don't log to wandb)
                prompt_filename = string_to_filename(prompt)[:25]
                hash_suffix = hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]
                video_filename = f"validation-real-{step}-{i}-{prompt_filename}-{hash_suffix}.mp4"
                validation_path = self.args.output_dir / "validation_res"
                video_path = str(validation_path / video_filename)
                
                logger.info(f"Process {accelerator.process_index}, i: {i}, Saving real video to {video_path}", main_process_only=False)
                export_to_video(video, video_path, fps=self.args.gen_fps)
                real_counts = count_first_last_nuclei(video_path)

            # Run the validation step and handle generated artifacts
            logger.info(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            
            validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe)
            # # Only main process runs the validation step
            # if self.accelerator.is_main_process:
            #     validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe)
            # else:
            #     validation_artifacts = []

            # Skip processing of validation_artifacts for non-main process in zero_stage == 3
            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            # Now process all generated artifacts from validation_step
            for k, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                if artifact_type not in ["video"] or artifact_value is None:
                    continue
                    
                # Generate a unique filename for the artifact
                prompt_filename = string_to_filename(prompt)[:25]
                hash_suffix = hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]
                extension = "mp4"
                gen_filename = f"validation-gen-{step}-{i}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                gen_path = str(validation_path / gen_filename)
                
 
                logger.info(f"Process {accelerator.process_index}, i: {i}, Saving generated video to {gen_path}", main_process_only=False)
                export_to_video(artifact_value, gen_path, fps=self.args.gen_fps)
                
                # Extract and save first frame
                if len(artifact_value) > 0:
                    first_frame = artifact_value[0]
                    first_frame_filename = gen_path.replace(".mp4", "_first_frame.png")
                    first_frame.save(first_frame_filename)
                    first_frame_wandb = wandb.Image(
                        first_frame_filename, 
                        caption=f"Sample {i} - First frame - {prompt}"
                    )
                    all_processes_artifacts.append(first_frame_wandb)
                    
                    # Calculate similarity metrics if we have a conditioning image
                    if image is not None:
                        img1 = np.array(image)
                        img2 = np.array(first_frame)
                        
                        if img1.shape == img2.shape:
                            # Calculate MSE
                            mse_value = np.mean((img1 - img2) ** 2)
                            logger.info(f"Process {accelerator.process_index}, i: {i},  MSE between conditioning image and first frame: {mse_value:.4f}", main_process_only=False)
                            # Add to artifacts instead of direct logging
                            all_processes_artifacts.append({
                                "type": "metric",
                                "name": f"val_image_similarity/mse_sample{i}",
                                "value": mse_value
                            })
                            
                            # Calculate SSIM
                            img1_gray = np.array(image.convert("L"))
                            img2_gray = np.array(first_frame.convert("L"))
                            ssim_value = ssim(img1_gray, img2_gray)
                            logger.info(f"Process {accelerator.process_index}, i: {i},  SSIM between conditioning image and first frame: {ssim_value:.4f}", main_process_only=False)
                            # Add to artifacts instead of direct logging
                            all_processes_artifacts.append({
                                "type": "metric",
                                "name": f"val_image_similarity/ssim_sample{i}",
                                "value": ssim_value
                            })
                    
                    # Log the video to wandb
                    video_wandb = wandb.Video(gen_path, caption=f"Sample {i} - {prompt}")
                    all_processes_artifacts.append(video_wandb)
                    
                    # Calculate nuclei counts
                    gen_counts = count_first_last_nuclei(gen_path)
                    
                    # Log comparison between real and generated videos
                    if real_counts is not None and real_counts[2] is not None and gen_counts[2] is not None:
                        count_first_real, count_last_real, ratio_real = real_counts
                        count_first_gen, count_last_gen, ratio_gen = gen_counts
                        
                        logger.info(
                            f"Process {accelerator.process_index}, i: {i}, [Val@step={step}] Prompt: {prompt}\n"
                            f"  Real video:  count_first={count_first_real}, count_last={count_last_real}, ratio={ratio_real:.2f}\n"
                            f"  Synth video: count_first={count_first_gen}, count_last={count_last_gen}, ratio={ratio_gen:.2f}",
                            main_process_only=False
                        )

                        # For ratio_of_ratios
                        if ratio_real > 0:
                            ratio_of_ratios = ratio_gen / ratio_real
                            logger.info(f"Process {accelerator.process_index}, i: {i},  ratio_of_ratios = {ratio_of_ratios:.2f}", main_process_only=False)
                            # Add to artifacts instead of direct logging
                            all_processes_artifacts.append({
                                "type": "metric",
                                "name": f"val_ratio_of_ratios_sample{i}",
                                "value": ratio_of_ratios
                            })


        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            
            # Add debug logging
            logger.info(f"Main process preparing to log artifacts. all_artifacts length: {len(all_artifacts)}")
            
            # Extract metrics from all_artifacts
            metrics_dict = {}
            image_artifacts = []
            video_artifacts = []
            
            for artifact in all_artifacts:
                if isinstance(artifact, dict) and artifact.get("type") == "metric":
                    metrics_dict[artifact["name"]] = artifact["value"]
                elif isinstance(artifact, wandb.Image):
                    image_artifacts.append(artifact)
                elif isinstance(artifact, wandb.Video):
                    video_artifacts.append(artifact)
            
            logger.info(f"Extracted {len(metrics_dict)} metrics, {len(image_artifacts)} images, {len(video_artifacts)} videos")
            
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    logger.info(f"Logging to wandb: {len(video_artifacts)} videos, {len(image_artifacts)} images")
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                            **metrics_dict
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)
            
        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch, pipe) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)
