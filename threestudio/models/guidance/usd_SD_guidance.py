import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("usd-text-to-3D_guidance")
class USDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        self.submodules = SubModules(pipe=pipe)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe.scheduler = self.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_usd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            
            noise_pred = self.forward_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
            )

        noise_pred_text, noise_pred_pretrain_uncond = noise_pred.chunk(2)
        # lambda == 0
        delta_noise_pred = self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_pretrain_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (delta_noise_pred)
        return grad

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents, rgb_BCHW_512

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents, rgb_BCHW_512 = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(elevation, azimuth, camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )

        grad = self.compute_grad_usd(
            latents, 
            text_embeddings_vd, 
        )
        grad = torch.nan_to_num(grad)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_usd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        loss_dict = {
            "loss_usd": loss_usd,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return loss_dict

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )