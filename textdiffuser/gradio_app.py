# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
import re
import zipfile

if not os.path.exists('textdiffuser-ckpt'):
    os.system('wget https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/textdiffuser-ckpt-new.zip')
    with zipfile.ZipFile('textdiffuser-ckpt-new.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

if not os.path.exists('images'):
    os.system('wget https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/images.zip')
    with zipfile.ZipFile('images.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

import cv2
import random
import logging
import argparse
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional
from packaging import version
from termcolor import colored
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance # import for visualization
from huggingface_hub import HfFolder, Repository, create_repo, whoami

import datasets
from datasets import load_dataset
from datasets import disable_caching

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel 
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from util import segmentation_mask_visualization, make_caption_pil, combine_image, combine_image_gradio, transform_mask, transform_mask_pil, filter_segmentation_mask, inpainting_merge_image
from model.layout_generator import get_layout_from_prompt
from model.text_segmenter.unet import UNet


disable_caching()
check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str,
        default='runwayml/stable-diffusion-v1-5', # no need to modify this  
        help="Path to pretrained model or model identifier from huggingface.co/models. Please do not modify this.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text-to-image",
        # required=True,
        choices=["text-to-image", "text-to-image-with-template", "text-inpainting"],
        help="Three modes can be used.",
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        default="",
        # required=True,
        help="The text prompts provided by users.",
    )
    parser.add_argument(
        "--template_image", 
        type=str,
        default="",
        help="The template image should be given when using 【text-to-image-with-template】 mode.",
    )
    parser.add_argument(
        "--original_image", 
        type=str,
        default="",
        help="The original image should be given when using 【text-inpainting】 mode.",
    )
    parser.add_argument(
        "--text_mask", 
        type=str,
        default="",
        help="The text mask should be given when using 【text-inpainting】 mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--classifier_free_scale", 
        type=float,
        default=7.5, # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Classifier free scale following https://arxiv.org/abs/2207.12598.",
    )
    parser.add_argument(
        "--drop_caption", 
        action="store_true", 
        help="Whether to drop captions during training following https://arxiv.org/abs/2207.12598.."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0, 
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to", 
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500, 
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='textdiffuser-ckpt/diffusion_backbone', # should be specified during inference
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--font_path", 
        type=str, 
        default='Arial.ttf', 
        help="The path of font for visualization."
    )
    parser.add_argument(
        "--sample_steps", 
        type=int, 
        default=50, # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Diffusion steps for sampling."
    )
    parser.add_argument(
        "--vis_num", 
        type=int, 
        default=4, # please decreases the number if out-of-memory error occurs
        help="Number of images to be sample. Please decrease it when encountering out of memory error."
    )
    parser.add_argument(
        "--binarization", 
        action="store_true", 
        help="Whether to binarize the template image."
    )
    parser.add_argument(
        "--use_pillow_segmentation_mask", 
        type=bool,
        default=True, 
        help="In the 【text-to-image】 mode, please specify whether to use the segmentation masks provided by PILLOW"
    )
    parser.add_argument(
        "--character_segmenter_path", 
        type=str,
        default='textdiffuser-ckpt/text_segmenter.pth',
        help="checkpoint of character-level segmenter"
    )
    args = parser.parse_args()
    
    print(f'{colored("[√]", "green")} Arguments are loaded.')
    print(args)
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



args = parse_args()
logging_dir = os.path.join(args.output_dir, args.logging_dir)

print(f'{colored("[√]", "green")} Logging dir is set to {logging_dir}.')

accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    logging_dir=logging_dir,
    project_config=accelerator_project_config,
)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# Handle the repository creation
if accelerator.is_main_process:
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        print(args.output_dir)

# Load scheduler, tokenizer and models.
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision).cuda()
unet = UNet2DConditionModel.from_pretrained(
    args.resume_from_checkpoint, subfolder="unet", revision=None 
).cuda() 

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        
        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


# setup schedulers                    
scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
# sample_num = args.vis_num

def to_tensor(image):
    if isinstance(image, Image.Image):  
        image = np.array(image)
    elif not isinstance(image, np.ndarray):  
        raise TypeError("Error")

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image)

    return tensor

def text_to_image(prompt,slider_step,slider_guidance,slider_batch):

    prompt = prompt.replace('"', "'")
    prompt = re.sub(r"[^a-zA-Z0-9'\" ]+", "", prompt)

    if slider_step>=100:
        slider_step = 100
        
    args.prompt = prompt 
    sample_num = slider_batch
    seed = random.randint(0, 10000000)
    set_seed(seed)
    scheduler.set_timesteps(slider_step) 

    noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
    input = noise # (b, 4, 64, 64)

    captions = [args.prompt] * sample_num
    captions_nocond = [""] * sample_num
    print(f'{colored("[√]", "green")} Prompt is loaded: {args.prompt}.')
    
    # encode text prompts
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states = text_encoder(inputs)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states: {encoder_hidden_states.shape}.')

    inputs_nocond = tokenizer(
        captions_nocond, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states_nocond = text_encoder(inputs_nocond)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.')

    # load character-level segmenter
    segmenter = UNet(3, 96, True).cuda()
    segmenter = torch.nn.DataParallel(segmenter)
    segmenter.load_state_dict(torch.load(args.character_segmenter_path))
    segmenter.eval()
    print(f'{colored("[√]", "green")} Text segmenter is successfully loaded.')

    #### text-to-image ####
    render_image, segmentation_mask_from_pillow = get_layout_from_prompt(args)
    
    segmentation_mask = torch.Tensor(np.array(segmentation_mask_from_pillow)).cuda() # (512, 512)

    segmentation_mask = filter_segmentation_mask(segmentation_mask)
    segmentation_mask = torch.nn.functional.interpolate(segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest')
    segmentation_mask = segmentation_mask.squeeze(1).repeat(sample_num, 1, 1).long().to('cuda') # (1, 1, 256, 256)
    print(f'{colored("[√]", "green")} character-level segmentation_mask: {segmentation_mask.shape}.')
    
    feature_mask = torch.ones(sample_num, 1, 64, 64).to('cuda') # (b, 1, 64, 64)
    masked_image = torch.zeros(sample_num, 3, 512, 512).to('cuda') # (b, 3, 512, 512)
    masked_feature = vae.encode(masked_image).latent_dist.sample() # (b, 4, 64, 64)
    masked_feature = masked_feature * vae.config.scaling_factor 
    print(f'{colored("[√]", "green")} feature_mask: {feature_mask.shape}.')
    print(f'{colored("[√]", "green")} masked_feature: {masked_feature.shape}.')

    # diffusion process
    intermediate_images = []
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample 
            input = prev_noisy_sample
            intermediate_images.append(prev_noisy_sample)
            
    # decode and visualization
    input = 1 / vae.config.scaling_factor * input 
    sample_images = vae.decode(input.float(), return_dict=False)[0] # (b, 3, 512, 512)

    image_pil = render_image.resize((512,512))
    segmentation_mask = segmentation_mask[0].squeeze().cpu().numpy()
    character_mask_pil = Image.fromarray(((segmentation_mask!=0)*255).astype('uint8')).resize((512,512))
    character_mask_highlight_pil = segmentation_mask_visualization(args.font_path,segmentation_mask)
    caption_pil = make_caption_pil(args.font_path, captions)
    
    # save pred_img
    pred_image_list = []
    for image in sample_images.float():
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
        pred_image_list.append(image)
        
    blank_pil = combine_image_gradio(args, None, pred_image_list, image_pil, character_mask_pil, character_mask_highlight_pil, caption_pil)
    
    intermediate_result = Image.new('RGB', (512*3, 512))
    intermediate_result.paste(image_pil, (0, 0))
    intermediate_result.paste(character_mask_pil, (512, 0))
    intermediate_result.paste(character_mask_highlight_pil, (512*2, 0))

    return blank_pil, intermediate_result


# load character-level segmenter
segmenter = UNet(3, 96, True).cuda()
segmenter = torch.nn.DataParallel(segmenter)
segmenter.load_state_dict(torch.load(args.character_segmenter_path))
segmenter.eval()
print(f'{colored("[√]", "green")} Text segmenter is successfully loaded.')




def text_to_image_with_template(prompt,template_image,slider_step,slider_guidance,slider_batch, binary):

    if slider_step>=100:
        slider_step = 100
        
    orig_template_image = template_image.resize((512,512)).convert('RGB')
    args.prompt = prompt 
    sample_num = slider_batch
    # If passed along, set the training seed now.
    # seed = slider_seed
    seed = random.randint(0, 10000000)
    set_seed(seed)
    scheduler.set_timesteps(slider_step) 

    noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
    input = noise # (b, 4, 64, 64)

    captions = [args.prompt] * sample_num
    captions_nocond = [""] * sample_num
    print(f'{colored("[√]", "green")} Prompt is loaded: {args.prompt}.')
    
    # encode text prompts
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states = text_encoder(inputs)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states: {encoder_hidden_states.shape}.')

    inputs_nocond = tokenizer(
        captions_nocond, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states_nocond = text_encoder(inputs_nocond)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.')

    #### text-to-image-with-template ####
    template_image = template_image.resize((256,256)).convert('RGB')
    
    # whether binarization is needed
    print(f'{colored("[Warning]", "red")} args.binarization is set to {binary}. You may need it when using handwritten images as templates.')
        
    if binary:
        gray = ImageOps.grayscale(template_image)
        binary = gray.point(lambda x: 255 if x > 96 else 0, '1')
        template_image = binary.convert('RGB')

    # to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(template_image).unsqueeze(0).cuda().sub_(0.5).div_(0.5) # (b, 3, 256, 256)
            
    with torch.no_grad():
        segmentation_mask = segmenter(image_tensor) # (b, 96, 256, 256)
    segmentation_mask = segmentation_mask.max(1)[1].squeeze(0) # (256, 256)
    segmentation_mask = filter_segmentation_mask(segmentation_mask) # (256, 256)
    
    segmentation_mask = torch.nn.functional.interpolate(segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest') # (b, 1, 256, 256)
    segmentation_mask = segmentation_mask.squeeze(1).repeat(sample_num, 1, 1).long().to('cuda') # (b, 1, 256, 256)
    print(f'{colored("[√]", "green")} Character-level segmentation_mask: {segmentation_mask.shape}.')
    
    feature_mask = torch.ones(sample_num, 1, 64, 64).to('cuda') # (b, 1, 64, 64)
    masked_image = torch.zeros(sample_num, 3, 512, 512).to('cuda') # (b, 3, 512, 512)
    masked_feature = vae.encode(masked_image).latent_dist.sample() # (b, 4, 64, 64)
    masked_feature = masked_feature * vae.config.scaling_factor # (b, 4, 64, 64)

    # diffusion process
    intermediate_images = []
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample 
            input = prev_noisy_sample
            intermediate_images.append(prev_noisy_sample)
            
    # decode and visualization
    input = 1 / vae.config.scaling_factor * input 
    sample_images = vae.decode(input.float(), return_dict=False)[0] # (b, 3, 512, 512)

    image_pil = None
    segmentation_mask = segmentation_mask[0].squeeze().cpu().numpy()
    character_mask_pil = Image.fromarray(((segmentation_mask!=0)*255).astype('uint8')).resize((512,512))
    character_mask_highlight_pil = segmentation_mask_visualization(args.font_path,segmentation_mask)
    caption_pil = make_caption_pil(args.font_path, captions)
    
    # save pred_img
    pred_image_list = []
    for image in sample_images.float():
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
        pred_image_list.append(image)
        
    blank_pil = combine_image_gradio(args, None, pred_image_list, image_pil, character_mask_pil, character_mask_highlight_pil, caption_pil)
    
    intermediate_result = Image.new('RGB', (512*3, 512))
    intermediate_result.paste(orig_template_image, (0, 0))
    intermediate_result.paste(character_mask_pil, (512, 0))
    intermediate_result.paste(character_mask_highlight_pil, (512*2, 0))
    
    return blank_pil, intermediate_result


def text_inpainting(prompt,orig_image,mask_image,slider_step,slider_guidance,slider_batch):

    if slider_step>=100:
        slider_step = 100
        
    args.prompt = prompt 
    sample_num = slider_batch
    # If passed along, set the training seed now.
    # seed = slider_seed
    seed = random.randint(0, 10000000)
    set_seed(seed)
    scheduler.set_timesteps(slider_step) 

    noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
    input = noise # (b, 4, 64, 64)

    captions = [args.prompt] * sample_num
    captions_nocond = [""] * sample_num
    print(f'{colored("[√]", "green")} Prompt is loaded: {args.prompt}.')
    
    # encode text prompts
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states = text_encoder(inputs)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states: {encoder_hidden_states.shape}.')

    inputs_nocond = tokenizer(
        captions_nocond, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids # (b, 77)
    encoder_hidden_states_nocond = text_encoder(inputs_nocond)[0].cuda() # (b, 77, 768)
    print(f'{colored("[√]", "green")} encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.')

    mask_image = cv2.resize(mask_image, (512,512))
    # mask_image = mask_image.resize((512,512)).convert('RGB')
    text_mask = np.array(mask_image)
    threshold = 128  
    _, text_mask = cv2.threshold(text_mask, threshold, 255, cv2.THRESH_BINARY)
    text_mask = Image.fromarray(text_mask).convert('RGB').resize((256,256))
    text_mask.save('text_mask.png') 
    text_mask_tensor = to_tensor(text_mask).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
    with torch.no_grad():
        segmentation_mask = segmenter(text_mask_tensor)
        
    segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)
    segmentation_mask = filter_segmentation_mask(segmentation_mask)
    segmentation_mask = torch.nn.functional.interpolate(segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest')

    image_mask = transform_mask_pil(mask_image) 
    image_mask = torch.from_numpy(image_mask).cuda().unsqueeze(0).unsqueeze(0) 

    orig_image = orig_image.convert('RGB').resize((512,512))
    image = orig_image
    image_tensor = to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)   
    masked_image = image_tensor * (1-image_mask)
    masked_feature = vae.encode(masked_image).latent_dist.sample().repeat(sample_num, 1, 1, 1)
    masked_feature = masked_feature * vae.config.scaling_factor
    
    image_mask = torch.nn.functional.interpolate(image_mask, size=(256, 256), mode='nearest').repeat(sample_num, 1, 1, 1) 
    segmentation_mask = segmentation_mask * image_mask 
    feature_mask = torch.nn.functional.interpolate(image_mask, size=(64, 64), mode='nearest')

    # diffusion process
    intermediate_images = []
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample 
            input = prev_noisy_sample
            intermediate_images.append(prev_noisy_sample)
            
    # decode and visualization
    input = 1 / vae.config.scaling_factor * input 
    sample_images = vae.decode(input.float(), return_dict=False)[0] # (b, 3, 512, 512)

    image_pil = None
    segmentation_mask = segmentation_mask[0].squeeze().cpu().numpy()
    character_mask_pil = Image.fromarray(((segmentation_mask!=0)*255).astype('uint8')).resize((512,512))
    character_mask_highlight_pil = segmentation_mask_visualization(args.font_path,segmentation_mask)
    caption_pil = make_caption_pil(args.font_path, captions)
    
    # save pred_img
    pred_image_list = []
    for image in sample_images.float():
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
        
        # need to merge
        
        # image = inpainting_merge_image(orig_image, Image.fromarray(mask_image).convert('L'), image)

        pred_image_list.append(image)
    
    character_mask_pil.save('character_mask_pil.png')
    character_mask_highlight_pil.save('character_mask_highlight_pil.png')
    
        
    blank_pil = combine_image_gradio(args, None, pred_image_list, image_pil, character_mask_pil, character_mask_highlight_pil, caption_pil)


    background = orig_image.resize((512, 512))
    alpha = Image.new('L', background.size, int(255 * 0.2))
    background.putalpha(alpha)
    # foreground
    foreground = Image.fromarray(mask_image).convert('L').resize((512, 512))
    threshold = 200
    alpha = foreground.point(lambda x: 0 if x > threshold else 255, '1')
    foreground.putalpha(alpha)
    merge_image = Image.alpha_composite(foreground.convert('RGBA'), background.convert('RGBA')).convert('RGB')

    intermediate_result = Image.new('RGB', (512*3, 512))
    intermediate_result.paste(merge_image, (0, 0))
    intermediate_result.paste(character_mask_pil, (512, 0))
    intermediate_result.paste(character_mask_highlight_pil, (512*2, 0))
    
    return blank_pil, intermediate_result
    
import gradio as gr
    
with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            TextDiffuser: Diffusion Models as Text Painters
        </h1>        
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
        [<a href="https://arxiv.org/abs/2305.10855" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/microsoft/unilm/tree/master/textdiffuser" style="color:blue;">Code</a>]
        [<a href="https://jingyechen.github.io/textdiffuser/" style="color:blue;">ProjectPage</a>]
        </h3> 
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We propose <b>TextDiffuser</b>, a flexible and controllable framework to generate images with visually appealing text that is coherent with backgrounds. 
        Main features include: (a) <b><font color="#A52A2A">Text-to-Image</font></b>: The user provides a prompt and encloses the keywords with single quotes (e.g., a text image of ‘hello’). The model first determines the layout of the keywords and then draws the image based on the layout and prompt. (b) <b><font color="#A52A2A">Text-to-Image with Templates</font></b>: The user provides a prompt and a template image containing text, which can be a printed, handwritten, or scene text image. These template images can be used to determine the layout of the characters. (c) <b><font color="#A52A2A">Text Inpainting</font></b>: The user provides an image and specifies the region to be modified along with the desired text content. The model is able to modify the original text or add text to areas without text.
        </h2>
        <img src="file/images/huggingface_blank.jpg" alt="textdiffuser">        
        </div>
        """)

    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Input your prompt here. Please enclose keywords with 【single quotes】, you may refer to the examples below. The current version only supports input in English characters.", placeholder="Placeholder 'Team' hat")
                slider_step = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling step", info="The sampling step for TextDiffuser.")
                slider_guidance = gr.Slider(minimum=1, maximum=9, value=7.5, step=0.5, label="Scale of classifier-free guidance", info="The scale of classifier-free guidance and is set to 7.5 in default.")
                slider_batch = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Batch size", info="The number of images to be sampled.")
                # slider_seed = gr.Slider(minimum=1, maximum=10000, label="Seed", randomize=True)
                button = gr.Button("Generate")
                            
            with gr.Column(scale=1):
                output = gr.Image(label='Generated image')
                
                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("Layout, segmentation mask, and details of segmentation mask from left to right.")
                    intermediate_results = gr.Image(label='')
        
        gr.Markdown("## Prompt Examples")
        gr.Examples(
            [
                ["'Team' hat"],
                ["Thanksgiving 'Fam' Mens T Shirt"],
                ["A storefront with 'Hello World' written on it."],
                ["A poster titled 'Quails of North America', showing different kinds of quails."],
                ["A storefront with 'Deep Learning' written on it."],
                ["An antique bottle labeled 'Energy Tonic'"],
                ["A TV show poster titled 'Tango argentino'"],
                ["A TV show poster with logo 'The Dry' on it"],
                ["Stupid 'History' eBook Tales of Stupidity Strangeness"],
                ["Photos of 'Sampa Hostel'"],
                ["A cover named 'Anything is possible'"],
                ["A large recipe book titled 'Recipes from Peru'."],
                ["New York Skyline with 'Diffusion' written with fireworks on the sky"],
                ["Books with the word 'Science' printed on them"],
                ["A globe with the words 'Planet Earth' written in bold letters with continents in bright colors"],
                ["A logo for the company 'EcoGrow', where the letters look like plants"],
            ],
            prompt,
            examples_per_page=100
        )
                    
        button.click(text_to_image, inputs=[prompt,slider_step,slider_guidance,slider_batch], outputs=[output,intermediate_results])
        
    with gr.Tab("Text-to-Image-with-Template"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label='Input your prompt here.')
                template_image = gr.Image(label='Template image', type="pil")
                slider_step = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling step", info="The sampling step for TextDiffuser.")
                slider_guidance = gr.Slider(minimum=1, maximum=9, value=7.5, step=0.5, label="Scale of classifier-free guidance", info="The scale of classifier-free guidance and is set to 7.5 in default.")
                slider_batch = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Batch size", info="The number of images to be sampled.")
                # binary = gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?")
                binary = gr.Checkbox(label="Binarization", bool=True, info="Whether to binarize the template image? You may need it when using handwritten images as templates.")
                button = gr.Button("Generate")
                    
            with gr.Column(scale=1):
                output = gr.Image(label='Generated image')
                
                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("Template image, segmentation mask, and details of segmentation mask from left to right.")
                    intermediate_results = gr.Image(label='')

        gr.Markdown("## Prompt and Template-Image Examples")
        gr.Examples(
            [
                ["a hand-drawn blueprint for a time machine with the caption 'Time traveling device'", './images/text-to-image-with-template/5.jpg', False], 
                ["a gate of garden", './images/text-to-image-with-template/6.jpg', False], 
                ["a book called summer vibe written by diffusion model", './images/text-to-image-with-template/7.jpg', False], 
                ["a work company", './images/text-to-image-with-template/8.jpg', False], 
                ["a book of AI in next century written by AI robot ", './images/text-to-image-with-template/9.jpg', False], 
                ["A board saying having a dog named shark at the beach was a mistake", './images/text-to-image-with-template/1.jpg', False], 
                ["an elephant holds a newspaper that is written elephant take over the world", './images/text-to-image-with-template/2.jpg', False], 
                ["a mouse with a flashlight saying i am afraid of the dark", './images/text-to-image-with-template/4.jpg', False], 
                ["a birthday cake of happy birthday to xyz", './images/text-to-image-with-template/10.jpg', False], 
                ["a poster of monkey music festival", './images/text-to-image-with-template/11.jpg', False], 
                ["a meme of are you kidding", './images/text-to-image-with-template/12.jpg', False], 
                ["a 3d model of a 1980s-style computer with the text my old habit on the screen", './images/text-to-image-with-template/13.jpg', True], 
                ["a board of hello world", './images/text-to-image-with-template/15.jpg', True], 
                ["a microsoft bag", './images/text-to-image-with-template/16.jpg', True], 
                ["a dog holds a paper saying please adopt me", './images/text-to-image-with-template/17.jpg', False], 
                ["a hello world banner", './images/text-to-image-with-template/18.jpg', False], 
                ["a stop pizza", './images/text-to-image-with-template/19.jpg', False], 
                ["a dress with text do not read the next sentence", './images/text-to-image-with-template/20.jpg', False], 
            ],
            [prompt,template_image, binary],
            examples_per_page=100
        )

        button.click(text_to_image_with_template, inputs=[prompt,template_image,slider_step,slider_guidance,slider_batch,binary], outputs=[output,intermediate_results])
        
    with gr.Tab("Text-Inpainting"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label='Input your prompt here.')
                with gr.Row():
                    orig_image = gr.Image(label='Original image', type="pil")
                    mask_image = gr.Image(label='Mask image', type="numpy")
                slider_step = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling step", info="The sampling step for TextDiffuser.")
                slider_guidance = gr.Slider(minimum=1, maximum=9, value=7.5, step=0.5, label="Scale of classifier-free guidance", info="The scale of classifier-free guidance and is set to 7.5 in default.")
                slider_batch = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Batch size", info="The number of images to be sampled.")
                button = gr.Button("Generate")
            with gr.Column(scale=1):
                output = gr.Image(label='Generated image')
                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("Masked image, segmentation mask, and details of segmentation mask from left to right.")
                    intermediate_results = gr.Image(label='')
                
        gr.Markdown("## Prompt, Original Image, and Mask Examples")
        gr.Examples(
            [
                ["eye on security protection", './images/text-inpainting/1.jpg', './images/text-inpainting/1mask.jpg'],
                ["a logo of poppins", './images/text-inpainting/2.jpg', './images/text-inpainting/2mask.jpg'],
                ["tips for middle space living ", './images/text-inpainting/3.jpg', './images/text-inpainting/3mask.jpg'],
                ["george is a proud big sister", './images/text-inpainting/5.jpg', './images/text-inpainting/5mask.jpg'],
                ["we are the great people", './images/text-inpainting/6.jpg', './images/text-inpainting/6mask.jpg'],
                ["tech house interesting terrace party", './images/text-inpainting/7.jpg', './images/text-inpainting/7mask.jpg'],
                ["2023", './images/text-inpainting/8.jpg', './images/text-inpainting/8mask.jpg'],
                ["wear protective equipment necessary", './images/text-inpainting/9.jpg', './images/text-inpainting/9mask.jpg'],
                ["a good day in the hometown", './images/text-inpainting/10.jpg', './images/text-inpainting/10mask.jpg'],
                ["a boy paints good morning on a board", './images/text-inpainting/11.jpg', './images/text-inpainting/11mask.jpg'],
                ["the word my gift on a basketball", './images/text-inpainting/13.jpg', './images/text-inpainting/13mask.jpg'],
                ["a logo of mono", './images/text-inpainting/14.jpg', './images/text-inpainting/14mask.jpg'],
                ["a board saying assyrian on unflagging fry devastates", './images/text-inpainting/15.jpg', './images/text-inpainting/15mask.jpg'],
                ["a board saying session", './images/text-inpainting/16.jpg', './images/text-inpainting/16mask.jpg'],
                ["rankin dork", './images/text-inpainting/17mask.jpg', './images/text-inpainting/17.jpg'],
                ["a coin of mem", './images/text-inpainting/18mask.jpg', './images/text-inpainting/18.jpg'],
                ["a board without text", './images/text-inpainting/19.jpg', './images/text-inpainting/19mask.jpg'],
                ["a board without text", './images/text-inpainting/20.jpg', './images/text-inpainting/20mask.jpg'],

            ],
            [prompt,orig_image,mask_image],
        )
                
                
        button.click(text_inpainting, inputs=[prompt,orig_image,mask_image,slider_step,slider_guidance,slider_batch], outputs=[output, intermediate_results])



    gr.HTML(
        """
        <div style="text-align: justify; max-width: 1200px; margin: 20px auto;">
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Version</b>: 1.0
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Contact</b>: 
        For help or issues using TextDiffuser, please email Jingye Chen <a href="mailto:qwerty.chen@connect.ust.hk">(qwerty.chen@connect.ust.hk)</a>, Yupan Huang <a href="mailto:huangyp28@mail2.sysu.edu.cn">(huangyp28@mail2.sysu.edu.cn)</a> or submit a GitHub issue. For other communications related to TextDiffuser, please contact Lei Cui <a href="mailto:lecu@microsoft.com">(lecu@microsoft.com)</a> or Furu Wei <a href="mailto:fuwei@microsoft.com">(fuwei@microsoft.com)</a>.
        </h3>
        </div>
        """
    )

demo.launch()
