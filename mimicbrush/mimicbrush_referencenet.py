# Modified from https://github.com/tencent-ailab/IP-Adapter
import os
from typing import List
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from .utils import is_torch2_available
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor
from .resampler import LinearResampler



class MimicBrush_RefNet:
    def __init__(self, sd_pipe, image_encoder_path, model_ckpt, depth_estimator, depth_guider,referencenet, device):
        # Takes model path as input
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.model_ckpt = model_ckpt
        self.referencenet = referencenet.to(self.device)
        self.depth_estimator = depth_estimator.to(self.device).eval()
        self.depth_guider = depth_guider.to(self.device, dtype=torch.float16)
        self.pipe = sd_pipe.to(self.device)
        self.pipe.unet.set_attn_processor(AttnProcessor())

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        self.image_processor = VaeImageProcessor()
        self.load_checkpoint()

    def init_proj(self):
        image_proj_model = LinearResampler(
            input_dim=1280,
            output_dim=self.pipe.unet.config.cross_attention_dim,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_checkpoint(self):
        state_dict = torch.load(self.model_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.depth_guider.load_state_dict(state_dict["depth_guider"])
        print('=== load depth_guider ===')
        self.referencenet.load_state_dict(state_dict["referencenet"])
        print('=== load referencenet ===')
        self.image_encoder.load_state_dict(state_dict["image_encoder"])
        print('=== load image_encoder ===')
        if "unet" in state_dict.keys():
            self.pipe.unet.load_state_dict(state_dict["unet"])
            print('=== load unet ===')


    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds).to(dtype=torch.float16)
        
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def generate(
        self,
        pil_image=None,
        depth_image = None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=4,
        seed=None,
        image = None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        depth_image = depth_image.to(self.device)
        depth_map = self.depth_estimator(depth_image).unsqueeze(1)
        depth_feature = self.depth_guider(depth_map.to(self.device, dtype=torch.float16))

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=image_prompt_embeds , # image clip embedding 
            negative_prompt_embeds=uncond_image_prompt_embeds,  # uncond image clip embedding 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            referencenet=self.referencenet,
            source_image=pil_image,
            image = image,
            clip_image_embed= torch.cat([uncond_image_prompt_embeds, image_prompt_embeds], dim=0), # for reference U-Net
            depth_feature = depth_feature,
            **kwargs,
        ).images
        return images, depth_map



class MimicBrush_RefNet_inputmodel(MimicBrush_RefNet):
    # take model as input
    def __init__(self, sd_pipe, image_encoder, image_proj_model, depth_estimator, depth_guider, referencenet,  device):
        self.device = device
        self.image_encoder = image_encoder.to(
            self.device, dtype=torch.float16
        )
        self.depth_estimator = depth_estimator.to(self.device)
        self.depth_guider = depth_guider.to(self.device, dtype=torch.float16)
        self.image_proj_model = image_proj_model.to(self.device, dtype=torch.float16)
        self.referencenet = referencenet.to(self.device, dtype=torch.float16)
        self.pipe = sd_pipe.to(self.device)
        self.pipe.unet.set_attn_processor(AttnProcessor())
        self.referencenet.set_attn_processor(AttnProcessor())
        self.clip_image_processor = CLIPImageProcessor()
