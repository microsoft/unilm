# ------------------------------------------
# TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
# Paper Link: https://arxiv.org/abs/2311.16465
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser-2
# Copyright (c) Microsoft Corporation.
# ------------------------------------------

import os
import cv2
import random
import logging
import argparse
import numpy as np
import time

from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional
from packaging import version
from PIL import Image
from huggingface_hub import HfFolder, Repository, create_repo, whoami

import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

import datasets
from datasets import disable_caching

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision import transforms

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
        default=None, # should be specified during inference
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
    #### newly added parameters
    parser.add_argument(
        "--granularity", 
        type=int, 
        default=128, 
        help="The granularity of coordinates, ranging from 1~512."
    )
    parser.add_argument(
        "--coord_mode", 
        type=str, 
        default='lt',
        choices=['lt', 'center', 'ltrb'],
        help="The way to represent coordinates."
    )
    parser.add_argument(
        "--max_length", 
        default=77,
        type=int, 
        help="Maximum length of the composed prompt."
    )
    parser.add_argument(
        "--cfg", 
        default=7,
        type=float, 
        help="classifier free guidance."
    )
    parser.add_argument(
        "--sample_steps", 
        default=50,
        type=int, 
        help="steps for sampling for diffusion models."
    )
    parser.add_argument(
        "--input_format", 
        required=True,
        type=str, 
        help="specify the input format",
        choices=['prompt', 'prompts_txt_file', 'prompt_layout_txt_file']
    )
    parser.add_argument(
        "--input_prompt", 
        type=str, 
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
    )
    parser.add_argument(
        "--prompts_txt_file", 
        type=str, 
    )
    parser.add_argument(
        "--m1_model_path", 
        type=str, 
        help="the checkpoint of layout planner"
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=16,
        help=("The number of images to be visualized."),
    )
    args = parser.parse_args()
    
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


DATASET_NAME_MAPPING = {
    # "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "MARIO-10M": ("image", "text"), 
}

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    #### additional tokens are introduced, including coordinate tokens and character tokens
    print('***************')
    print(len(tokenizer))
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) 
    print(len(tokenizer))
    print('***************')

    if args.max_length == 77:
        text_encoder = CLIPTextModel.from_pretrained(
            args.resume_from_checkpoint, subfolder="text_encoder", ignore_mismatched_sizes=True
        )
    else:
        #### enlarge the context length of text encoder. empirically, enlarging the context length can proceed longer sequence. However, we observe that it will be hard to render general objects
        text_encoder = CLIPTextModel.from_pretrained(
            args.resume_from_checkpoint, subfolder="text_encoder", max_position_embeddings=args.max_length, ignore_mismatched_sizes=True
        )

    text_encoder.resize_token_embeddings(len(tokenizer))

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.resume_from_checkpoint, subfolder="unet"
    )
    # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)



    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                # model.save_pretrained(os.path.join(output_dir, "unet"))

                if i == 0:
                    model.save_pretrained(os.path.join(output_dir, f"unet"))
                elif i == 1:
                    model.save_pretrained(os.path.join(output_dir, f"text_encoder"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if i == 1:
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                elif i == 0:
                    load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder")
                    # model.register_to_config(**load_model.config)

                # # load diffusers style into model
                # load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                # model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype) 
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    
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


    # # Enable TF32 for faster training on Ampere GPUs,
    # # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if True:
    #     torch.backends.cuda.matmul.allow_tf32 = True

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(args.resume_from_checkpoint)

    if accelerator.is_main_process and os.path.exists(f'{args.output_dir}'):
        print('detect existing output_dir, removing the contained jpg/txt files ...')
        os.system(f'rm {args.output_dir}/*.jpg')
        os.system(f'rm {args.output_dir}/*.txt')

    # user_prompt = "Book cover of summer vibe, high quality, high resolution"
    # ocrs = [
    #     'Summer Vibe 20,20,100,40'       
    # ]
    
    if args.input_format == 'prompt_layout_txt_file':
        lines = open(args.input_file).readlines()
        user_prompts = [lines[0].strip()]
        ocrs = [lines[1:]]

    elif args.input_format == 'prompt' or args.input_format == 'prompts_txt_file':

        #### prepare m1 (layout planner)
        from fastchat.model import load_model, get_conversation_template
        m1_model, m1_tokenizer = load_model(
            args.m1_model_path,
            'cuda',
            1,
            None,
            False,
            False,
            revision="main",
            debug=False,
        )

        # prompt = 'a text image of hello world'
        prompts = []
        if args.input_format == 'prompt':
            prompts = [args.input_prompt]
        elif args.input_format == 'prompts_txt_file':
            prompts = open(args.prompts_txt_file).readlines()
        print(f'there are {len(prompts)} samples for generation')

        ocrs = []
        user_prompts = []
        for prompt in prompts:
            user_prompt = prompt
            user_prompts.append(user_prompt)
            template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
            msg = template
            conv = get_conversation_template(args.m1_model_path)
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = m1_tokenizer([prompt], return_token_type_ids=False)
            inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
            output_ids = m1_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.0,
                max_new_tokens=512,
            )

            if m1_model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = m1_tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            print(f"[{conv.roles[0]}]\n{msg}")
            print(f"[{conv.roles[1]}]\n{outputs}")
            # ocrs = outputs.split('\n')
            ocrs.append(outputs.split('\n'))


    with torch.no_grad():

        size = len(ocrs)
        print(f'the number of samples: {size}')

        time_seed = int(time.time())
        random.seed(time_seed)
        torch.manual_seed(time_seed)
        torch.cuda.manual_seed_all(time_seed)
  
        for sample_index in range(size):

            user_prompt = user_prompts[sample_index]
            current_ocr = ocrs[sample_index]

            ocr_ids = [] 
            print('user_prompt', user_prompt)
            print('current_ocr', current_ocr)

            current_ocr = []
            for ocr in current_ocr:
                ocr = ocr.strip()

                if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                    continue

                items = ocr.split()
                pred = ' '.join(items[:-1])
                box = items[-1]
            
                l,t,r,b = box.split(',')
                l,t,r,b = int(l), int(t), int(r), int(b)
                ocr_ids.extend(['l'+str(l), 't'+str(t), 'r'+str(r), 'b'+str(b)])

                char_list = list(pred)
                char_list = [f'[{i}]' for i in char_list]
                ocr_ids.extend(char_list)
                ocr_ids.append(tokenizer.eos_token_id)     
        
            caption_ids = tokenizer(
                user_prompt, truncation=True, return_tensors="pt"
            ).input_ids[0].tolist() 

            try:
                ocr_ids = tokenizer.encode(ocr_ids)
                prompt = caption_ids + ocr_ids
            except:
                prompt = caption_ids

            prompt = prompt[:args.max_length]
            while len(prompt) < args.max_length: 
                prompt.append(tokenizer.pad_token_id) 
            prompts_cond = prompt
            prompts_nocond = [tokenizer.pad_token_id]*args.max_length

            prompts_cond = [prompts_cond] * args.vis_num
            prompts_nocond = [prompts_nocond] * args.vis_num

            prompts_cond = torch.Tensor(prompts_cond).long().cuda()
            prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

            scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
            scheduler.set_timesteps(args.sample_steps) 
            noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda") 
            input = noise

            encoder_hidden_states_cond = text_encoder(prompts_cond)[0]
            encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0] 

            texts = prompts_cond
            f = open(f'{args.output_dir}/prompt_{sample_index}_{args.local_rank}.txt', 'w+')
            for text in texts:
                sentence = tokenizer.decode(text)
                f.write(sentence + '\n')
            f.close()

            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():  # classifier free guidance
                    noise_pred_cond = unet(sample=input.half(), timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:args.vis_num]).sample # b, 4, 64, 64
                    noise_pred_uncond = unet(sample=input.half(), timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:args.vis_num]).sample # b, 4, 64, 64
                    noisy_residual = noise_pred_uncond + args.cfg * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                    input = scheduler.step(noisy_residual, t, input).prev_sample
                    # input = prev_noisy_sample

            # decode
            input = 1 / vae.config.scaling_factor * input 
            images = vae.decode(input.half(), return_dict=False)[0] 
            width, height = 512, 512
            new_image = Image.new('RGB', (4*width, 4*height))
            for index, image in enumerate(images.float()):
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                row = index // 4
                col = index % 4
                new_image.paste(image, (col*width, row*height))
            new_image.save(f'{args.output_dir}/pred_img_{sample_index}_{args.local_rank}.jpg')

if __name__ == "__main__":
    main()
