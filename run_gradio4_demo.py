import gradio as gr
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file, load_file
from omegaconf import OmegaConf
from transformers import AutoConfig
import cv2
from PIL import Image
import numpy as np
import json
import os
#
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
#
from models.pipeline_mimicbrush import MimicBrushPipeline
from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from mimicbrush import MimicBrush_RefNet
from dataset.data_utils import *


val_configs = OmegaConf.load('./configs/inference.yaml')

# === import Depth Anything ===
import sys
sys.path.append("./depthanything")
from torchvision.transforms import Compose
from depthanything.fast_import import depth_anything_model 
from depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
depth_anything_model.load_state_dict(torch.load(val_configs.model_path.depth_model))



# === load the checkpoint ===
base_model_path = val_configs.model_path.pretrained_imitativer_path
vae_model_path = val_configs.model_path.pretrained_vae_name_or_path
image_encoder_path = val_configs.model_path.image_encoder_path
ref_model_path = val_configs.model_path.pretrained_reference_path
mimicbrush_ckpt = val_configs.model_path.mimicbrush_ckpt_path
device = "cuda"



def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size
    
    if height == width:
        return original_image
    
    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)
    
    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")
    
    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image


def collage_region(low, high, mask):
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8) 
    low = (low * 0).astype(np.uint8) 
    high = np.array(high).astype(np.uint8)
    mask_3 = mask 
    collage = low * mask_3 + high * (1-mask_3)
    collage = Image.fromarray(collage)
    return collage


def resize_image_keep_aspect_ratio(image, target_size = 512):
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def crop_padding_and_resize(ori_image, square_image):
    ori_height, ori_width, _ = ori_image.shape
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(square_image, (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)))
    padding_size = max(resized_square_image.shape[0] - ori_height, resized_square_image.shape[1] - ori_width)
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :,:]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right,:]
    return cropped_image


def vis_mask(image, mask):
    # mask 3 channle 255
    mask = mask[:,:,0]
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw outlines, using random colors
    outline_opacity = 0.5
    outline_thickness = 5
    outline_color = np.concatenate([ [255,255,255], [outline_opacity]  ])

    white_mask = np.ones_like(image) * 255

    mask_bin_3 = np.stack([mask,mask,mask],-1) > 128
    alpha = 0.5 
    image = ( white_mask * alpha + image * (1-alpha) ) * mask_bin_3 + image * (1-mask_bin_3)
    cv2.polylines(image, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)
    return image 



noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)


vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", in_channels=13, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=torch.float16)

pipe = MimicBrushPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    feature_extractor=None,
    safety_checker=None,
)

depth_guider = DepthGuider()
referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(dtype=torch.float16)
mimicbrush_model = MimicBrush_RefNet(pipe, image_encoder_path, mimicbrush_ckpt,  depth_anything_model, depth_guider, referencenet, device)
mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)


def infer_single(ref_image, target_image, target_mask, seed = -1, num_inference_steps=50, guidance_scale = 5, enable_shape_control = False):
    #return ref_image
    """
    mask: 0/1 1-channel  np.array
    image: rgb           np.array
    """

    ref_image = ref_image.astype(np.uint8)
    target_image = target_image.astype(np.uint8)
    target_mask  = target_mask .astype(np.uint8)

    ref_image = Image.fromarray(ref_image.astype(np.uint8)) 
    ref_image = pad_img_to_square(ref_image)

    target_image = pad_img_to_square(Image.fromarray(target_image))
    target_image_low = target_image


    target_mask = np.stack([target_mask,target_mask,target_mask],-1).astype(np.uint8) * 255
    target_mask_np = target_mask.copy()
    target_mask = Image.fromarray(target_mask) 
    target_mask = pad_img_to_square(target_mask, True)

    target_image_ori = target_image.copy()
    target_image = collage_region(target_image_low, target_image, target_mask)
    

    depth_image = target_image_ori.copy()
    depth_image = np.array(depth_image)
    depth_image = transform({'image': depth_image})['image']
    depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

    if not enable_shape_control:
        depth_image = depth_image * 0

    mask_pt = mask_processor.preprocess(target_mask, height=512, width=512)

    pred, depth_pred = mimicbrush_model.generate(pil_image=ref_image, depth_image = depth_image, num_samples=1, num_inference_steps=num_inference_steps,
                            seed=seed, image=target_image, mask_image=mask_pt, strength=1.0, guidance_scale=guidance_scale)


    depth_pred = F.interpolate(depth_pred, size=(512,512), mode = 'bilinear', align_corners=True)[0][0]
    depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
    depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
    depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:,:,::-1]

    pred = pred[0]
    pred = np.array(pred).astype(np.uint8)
    return pred, depth_pred.astype(np.uint8)



def inference_single_image(ref_image, 
                           tar_image, 
                           tar_mask, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           ):
    if seed == -1:
        seed = np.random.randint(10000)
    pred, depth_pred = infer_single(ref_image, tar_image, tar_mask, seed, num_inference_steps=ddim_steps, guidance_scale = scale, enable_shape_control = enable_shape_control)
    return pred, depth_pred



def run_local(base,
              ref,
              *args):
    image = base["background"].convert("RGB") #base["image"].convert("RGB")
    mask = base["layers"][0]  #base["mask"].convert("L")
    
    image = np.asarray(image)
    mask = np.asarray(mask)[:,:,-1]
    #print(image.shape, mask.shape, mask.max(), mask.min())
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    

    ref_image = ref.convert("RGB")
    ref_image = np.asarray(ref_image)

    if mask.sum() == 0:
        raise gr.Error('No mask for the background image.')
    
    mask_3 = np.stack([mask,mask,mask],-1).astype(np.uint8) * 255

    mask_alpha = mask_3.copy()
    for i in range(10):
        mask_alpha = cv2.GaussianBlur(mask_alpha, (3, 3), 0)
    
    synthesis, depth_pred = inference_single_image(ref_image.copy(), image.copy(), mask.copy(), *args)


    synthesis = crop_padding_and_resize(image, synthesis)
    depth_pred = crop_padding_and_resize(image, depth_pred)


    mask_3_bin = mask_alpha / 255
    synthesis = synthesis * mask_3_bin + image * (1-mask_3_bin)

    vis_source = vis_mask(image, mask_3).astype(np.uint8)
    return [synthesis.astype(np.uint8), depth_pred.astype(np.uint8), vis_source, mask_3]



with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("#  MimicBrush: Zero-shot Image Editing with Reference Imitation ")
        with gr.Row():
            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1, height=768)
            with gr.Accordion("Advanced Option", open=True):
                num_samples = 1
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=-30.0, maximum=30.0, value=5.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1)
                enable_shape_control = gr.Checkbox(label='Keep the original shape', value=False, interactive = True)
                
                gr.Markdown("### Tutorial")
                gr.Markdown("1. Upload the source image and the reference image")
                gr.Markdown("2. Select the \"draw button\" to mask the to-edit region on the source image  ")
                gr.Markdown("3. Click generate ")
                gr.Markdown("#### You shoud click \"keep the original shape\" to conduct texture transfer  ")
    
        gr.Markdown("# Upload the source image and reference image")
        gr.Markdown("### Tips: you could adjust the brush size")

        with gr.Row():
            base = gr.ImageEditor(  label="Source",
                                    type="pil",
                                    brush=gr.Brush(colors=["#000000"],default_size = 30, color_mode = "fixed"),
                                    layers = False,
                                    interactive=True
                                )
            ref = gr.Image(label="Reference", sources="upload", type="pil", height=512)
        run_local_button = gr.Button(value="Run")
        


    with gr.Row():
        gr.Examples(
        examples=[
            [
                './demo_example/005_source.png',
                './demo_example/005_reference.png', 
                0

            ],
            [
                './demo_example/004_source.png',
                './demo_example/004_reference.png', 
                0
            ],
   
            [
                './demo_example/000_source.png',
                './demo_example/000_reference.png', 
                0
            ],
            [
                './demo_example/003_source.png',
                './demo_example/003_reference.png', 
                0
            ],     

            [
                './demo_example/006_source.png',
                './demo_example/006_reference.png', 
                0
            ],
            [
                './demo_example/001_source.png',
                './demo_example/001_reference.png', 
                1
            ],
            [
                './demo_example/002_source.png',
                './demo_example/002_reference.png', 
                1
            ],

            [
                './demo_example/007_source.png',
                './demo_example/007_reference.png', 
                1
            ],
            [
                './demo_example/008_source.png',
                './demo_example/008_reference.png', 
                1
            ],
        ],

        inputs=[
                base,
                ref,
                enable_shape_control
                ],
                cache_examples=False,
                examples_per_page=100)


    run_local_button.click(fn=run_local, 
                           inputs=[base, 
                                   ref, 
                                   ddim_steps, 
                                   scale, 
                                   seed,
                                   enable_shape_control
                                   ], 
                           outputs=[baseline_gallery]
                        )

demo.launch(server_name="0.0.0.0")