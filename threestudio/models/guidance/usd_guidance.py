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

from extern.zero123 import Zero123Pipeline
import torchvision.transforms.functional as TF
import numpy as np

import imageio
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


@threestudio.register("usd-guidance")
class USDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        image_path: str = "../threestudio/images/cookies.png"
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

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.pipe.scheduler = self.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")
        
        image = imageio.imread(self.cfg.image_path) / 255.
        
        image = image[...,:3]*image[...,3:] + (1 - image[...,3:])
        self.image = torch.from_numpy(image)[None, ...].to(self.device).float()
        
        self.STEP = 0
        
        self.dtype = torch.float16
        self.zero123_pipe = Zero123Pipeline.from_pretrained(            
            '../threestudio/models/zero123-xl-diffusers',
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        self.zero123_pipe.image_encoder.eval()
        self.zero123_pipe.vae.eval()
        self.zero123_pipe.unet.eval()
        self.zero123_pipe.clip_camera_projection.eval()
        
        for param in self.zero123_pipe.vae.parameters():
            param.requires_grad = False 
        for param in self.zero123_pipe.unet.parameters():
            param.requires_grad = False 
        for param in self.zero123_pipe.image_encoder.parameters():
            param.requires_grad = False 
        for param in self.zero123_pipe.clip_camera_projection.parameters():
            param.requires_grad = False 
            
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        self.get_img_embeds(image)
        threestudio.info(f"Loaded Zero123!")
            

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
    def decode_latents(self, latents: Float[Tensor, "B 4 H W"], latent_height: int = 64, latent_width: int = 64,) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def decode_latents(self, latents):
        latents = 1 / self.zero123_pipe.vae.config.scaling_factor * latents
        imgs = self.zero123_pipe.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs, mode=False):
        imgs = 2 * imgs - 1
        
        posterior = self.zero123_pipe.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.zero123_pipe.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.zero123_pipe.feature_extractor(
            images=x_pil, return_tensors="pt"
        ).pixel_values.to(device=self.device, dtype=self.dtype)
        
        c = self.zero123_pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.zero123_pipe.vae.config.scaling_factor
        self.embeddings = [c, v]
    
    def compute_grad_usd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd,
        delta_elevation,
        delta_azimuth,
        delta_radius,
    ):
        batch_size = latents.shape[0]
        
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        
        noise = torch.randn_like(latents)
        
        with torch.no_grad():
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            
            T = np.stack([
                np.deg2rad([delta_elevation]), 
                np.sin(np.deg2rad([delta_azimuth])), 
                np.cos(np.deg2rad([delta_azimuth])), 
                [delta_radius]
            ], axis=-1)

            T = torch.from_numpy(T).unsqueeze(1).to(self.dtype).to(self.device) # [8, 1, 4]
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.zero123_pipe.clip_camera_projection(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

            noise_pred = self.zero123_pipe.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.zero123_pipe.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

        zero123_noise_pred_cond, zero123_noise_pred_uncond = noise_pred.chunk(2)
        zero123_noise_pred_cond = zero123_noise_pred_cond.float()
        zero123_noise_pred_uncond = zero123_noise_pred_uncond.float()
        
        
        with torch.no_grad():
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            noise_pred = self.forward_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
            )
            
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text = noise_pred_text.float()
        noise_pred_uncond = noise_pred_uncond.float()
        
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        
        grad = w * 7.5 * ((zero123_noise_pred_cond - noise_pred_uncond))
        
        return grad
    
    ###############################################################
    
    def get_latents(self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents
    
    def delta_noise(self, latents, text_embeddings, rgb_as_latents):
        ref_rgb_BCHW = self.image.permute(0, 3, 1, 2)
        ref_latents = self.get_latents(ref_rgb_BCHW, rgb_as_latents=rgb_as_latents)
        
        B = latents.shape[0]
        with torch.no_grad():
            
            t = torch.randint(
                20,
                980 + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            
            noise = torch.randn_like(latents)
            
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            
            ref_latents_noisy = self.scheduler.add_noise(ref_latents, noise, t)
            ref_latent_model_input = torch.cat([ref_latents_noisy] * 2, dim=0)
            ref_noise_pred = self.forward_unet(
                self.unet,
                ref_latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )
            ref_noise_pred_text, ref_noise_pred_uncond = ref_noise_pred.chunk(2)
            
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * 1 * (noise_pred_uncond - ref_noise_pred_uncond)
        
        return grad
    
    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        GT_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        
        delta_elevation,
        delta_azimuth,
        delta_radius,
        
        rgb_as_latents=False,
        **kwargs,
    ):
        # if self.STEP%10==0:
        #     torch.save(GT_rgb, 'hh.pt')
        #     torch.save(rgb, 'hh2.pt')
            
        batch_size = rgb.shape[0]
        
        loss = 0.
        
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW = F.interpolate(rgb_BCHW, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(rgb_BCHW.to(self.dtype))
        grad = self.compute_grad_usd(
            latents, 
            text_embeddings_vd, 
            delta_elevation, 
            delta_azimuth, 
            delta_radius
        )
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        latents = latents.float()
        target = target.float()
        loss_usd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss += loss_usd
        
        #######################################################################
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        
        GT_rgb_BCHW = GT_rgb.permute(0, 3, 1, 2)
        GT_latents = self.get_latents(GT_rgb_BCHW, rgb_as_latents=rgb_as_latents)
        ref_grad = self.delta_noise(GT_latents, text_embeddings, rgb_as_latents)
        ref_grad = torch.nan_to_num(ref_grad)
        GT_target = (GT_latents - ref_grad).detach()
        loss_ref = 0.5 * F.mse_loss(GT_latents, GT_target, reduction="sum") / batch_size
        loss += loss_ref
        
        self.STEP += 1
        
        return {
            "loss_vsd": loss,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )