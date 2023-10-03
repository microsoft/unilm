from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.utils import safe_getattr

DEFAULT_MAX_TARGET_POSITIONS = 1024

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
import inspect
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin
from diffusers.configuration_utils import FrozenDict


@register_model("diffusionmodel", dataclass=FairseqDataclass)
class Diffusionmodel(BaseFairseqModel, LoraLoaderMixin):
    def __init__(self, noise_scheduler, unet, scheduler, vae_scale_factor):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = vae_scale_factor

        self.config = FrozenDict([
            ('unet', ('diffusers', 'UNet2DConditionModel')),
            ('scheduler', ('diffusers', 'DPMSolverMultistepScheduler')),
        ])

    @property
    def components(self):
        components = {k: getattr(self, k) for k in self.config.keys()}
        return components

    @classmethod
    def build_model(cls, args, task, vae_scale_factor):
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16, revision="fp16"
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch.float16, revision="fp16"
        )
        if args.checkpoint_activations:
            unet.enable_gradient_checkpointing()
        if args.flash_attention:
            unet.enable_xformers_memory_efficient_attention()

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16, revision="fp16"
        )
        model = cls(noise_scheduler=noise_scheduler, unet=unet, scheduler=scheduler, vae_scale_factor=vae_scale_factor)
        return model

    def forward(self, latents, condition, null_token):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        # Predict the noise residual and compute loss
        random_p = torch.rand(bsz, device=latents.device)
        text_classifier_free_idx = random_p < 0.1
        condition[text_classifier_free_idx] = null_token[0]

        model_pred = self.unet(noisy_latents, timesteps, condition).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).sum()
        return {'diff_loss': loss}

    def sample(
            self,
            condition,
            height=512,
            width=512,
            num_inference_steps=50,
            text_guidance_scale=7.5,
            eta=0.0,
            lora_scale=0.0,
            **kwargs,
    ):
        batch_size = condition.shape[0]
        device = condition.device

        do_classifier_free_guidance = text_guidance_scale > 1.0
        if do_classifier_free_guidance:
            batch_size //= 2

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        shape = (batch_size, self.unet.config.in_channels, height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = randn_tensor(shape, device=device, dtype=condition.dtype)

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=condition,
                cross_attention_kwargs={"scale": lora_scale},
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents

    def sample_controlnet(
            self,
            condition,
            control_image,
            controlnet,
            num_inference_steps=50,
            text_guidance_scale=7.5,
            num_images_per_prompt=1,
            eta=0.0,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            **kwargs,
    ):
        control_guidance_start, control_guidance_end = [control_guidance_start], [control_guidance_end]

        if not hasattr(self, "control_image_processor"):
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
        batch_size = condition.shape[0]
        device = condition.device

        do_classifier_free_guidance = text_guidance_scale > 1.0
        if do_classifier_free_guidance:
            batch_size //= 2

        image = self.control_image_processor.preprocess(control_image).to(dtype=torch.float32)
        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        image = image.to(device=device, dtype=torch.float16)
        image = torch.cat([image] * 2)
        height, width = image.shape[-2:]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        shape = (batch_size, self.unet.config.in_channels, height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = randn_tensor(shape, device=device, dtype=condition.dtype)

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            control_model_input = latent_model_input
            controlnet_condition = condition

            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

            down_block_res_samples, mid_block_res_sample = controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_condition,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=condition,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents


@register_model("vae", dataclass=FairseqDataclass)
class VAE(BaseFairseqModel):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scaling_factor = 0.18215

    @classmethod
    def build_model(cls, args, task):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16, revision="fp16"
        )
        vae.requires_grad_(False)
        model = cls(vae=vae)
        return model

    def encode(self, x):
        self.vae.eval()
        return self.vae.encode(x).latent_dist.sample() * self.scaling_factor

    def encode_mode(self, x):
        self.vae.eval()
        return self.vae.encode(x).latent_dist.mode()

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def decode(self, latents, output_type="pil"):
        self.vae.eval()
        latents = 1 / self.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image


@register_model_architecture("diffusionmodel", "diffusionmodel_base")
def diffusionmodel_base(args):
    args.pretrained_model_name_or_path = safe_getattr(args, "pretrained_model_name_or_path",
                                                      "runwayml/stable-diffusion-v1-5")


@register_model_architecture("vae", "vae_base")
def vae_base(args):
    args.pretrained_model_name_or_path = safe_getattr(args, "pretrained_model_name_or_path",
                                                      "runwayml/stable-diffusion-v1-5")
