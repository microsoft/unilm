# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import argparse
import cv2
import torchvision.transforms as transforms

to_pil_image = transforms.ToPILImage()

def load_stablediffusion():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)      
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe

def test_stablediffusion(prompt, save_path, num_images_per_prompt=4,
                              pipe=None, generator=None):
    images = pipe(prompt, num_inference_steps=50, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))

def load_deepfloyd_if():
    from diffusers import DiffusionPipeline
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    # stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_1.enable_model_cpu_offload()
    stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16",
                                                torch_dtype=torch.float16)
    # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_2.enable_model_cpu_offload()
    safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker,
                      "watermarker": stage_1.watermarker}
    stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules,
                                                torch_dtype=torch.float16)
    # stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_3.enable_model_cpu_offload()
    return stage_1, stage_2, stage_3


def test_deepfloyd_if(stage_1, stage_2, stage_3, prompt, save_path, num_images_per_prompt=4, generator=None):
    idx = num_images_per_prompt - 1  # if the last image of a case exists, then return
    new_save_path = save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_' + str(idx) + '/')
    if os.path.exists(new_save_path):
        return
    if not stage_1 or not stage_2 or not stage_3:
        stage_1, stage_2, stage_3  = load_deepfloyd_if()
    if generator is None:
        generator = torch.manual_seed(0)
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    stage_1.set_progress_bar_config(disable=True)
    stage_2.set_progress_bar_config(disable=True)
    stage_3.set_progress_bar_config(disable=True)
    images = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                    output_type="pt",  num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image = stage_2(image=image.unsqueeze(0), prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                        generator=generator, output_type="pt").images
        image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
        # image = to_pil_image(image[0].cpu())
        new_save_path = save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/')
        image[0].save(new_save_path)


def load_controlnet_cannyedge():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,
                                                             safety_checker=None, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe


def test_controlnet_cannyedge(prompt, save_path, canny_path, num_images_per_prompt=4,
                              pipe=None, generator=None, low_threshold=100, high_threshold=200):
    '''ref: https://github.com/huggingface/diffusers/blob/131312caba0af97da98fc498dfdca335c9692f8c/docs/source/en/api/pipelines/stable_diffusion/controlnet.mdx'''
    from diffusers.utils import load_image
    if pipe is None:
        pipe = load_controlnet_cannyedge()

    if os.path.exists(canny_path):
        canny_path = Image.open(canny_path)
    image = load_image(canny_path)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    images = pipe(prompt, image, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))


def MARIOEval_generate_results(root, dataset, method='controlnet', num_images_per_prompt=4, split=0, total_split=1):
    root_eval = os.path.join(root, "MARIOEval")
    render_path = os.path.join(root_eval, dataset, 'render')
    root_res = os.path.join(root, "generation", method)
    for idx in range(num_images_per_prompt): 
        os.makedirs(os.path.join(root_res, dataset, 'images_' + str(idx)), exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    if method == 'controlnet':
        pipe = load_controlnet_cannyedge()
    elif method == 'stablediffusion':
        pipe = load_stablediffusion()
    elif method == 'deepfloyd':
        stage_1, stage_2, stage_3 = load_deepfloyd_if() 
   
    with open(os.path.join(root_eval, dataset, dataset + '.txt'), 'r') as fr:
        prompts = fr.readlines()
        prompts = [_.strip() for _ in prompts]
    for idx, prompt in tqdm(enumerate(prompts)): 
        if idx < split * len(prompts) / total_split or idx > (split + 1) * len(prompts) / total_split:
            continue
        if  method == 'controlnet':
            test_controlnet_cannyedge(prompt=prompt, num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  canny_path=os.path.join(render_path, str(idx) + '.png'),
                                  pipe=pipe, generator=generator) 
        elif  method == 'stablediffusion':
            test_stablediffusion(prompt=prompt, num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator) 
        elif method == 'deepfloyd':
            test_deepfloyd_if(stage_1, stage_2, stage_3, num_images_per_prompt=num_images_per_prompt,
                              save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                              prompt=prompt, generator=generator)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='TMDBEval500',
        required=False,
        choices=['TMDBEval500', 'OpenLibraryEval500', 'LAIONEval4000',
                 'ChineseDrawText', 'DrawBenchText', 'DrawTextCreative']
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/eval",
        required=True, 
    )
    parser.add_argument(
        "--method",
        type=str,
        default='controlnet', 
        required=False,
        choices=['controlnet', 'deepfloyd', 'stablediffusion', 'textdiffuser']
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--split", 
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--total_split", 
        type=int,
        default=1,
        required=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    MARIOEval_generate_results(root=args.root, dataset=args.dataset, method=args.method,
                               split=args.split, total_split=args.total_split) 
