# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
import cv2
import math
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from packaging import version
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import HfFolder, Repository, create_repo, whoami

import datasets
from datasets import load_dataset

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision import transforms

import transformers

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from termcolor import colored


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="sd-model-finetuned",
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
        "--character_aware_loss_lambda",
        type=float,
        default=0.01,
        help="Lambda for the character-aware loss",
    )
    parser.add_argument(
        "--character_aware_loss_ckpt", 
        type=str, 
        default='ckpt/character_aware_loss_unet.pth', 
        help="The checkpoint for unet providing the charactere-aware loss."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=2
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr", 
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--no_pos_con",
        action="store_true",
        default=False,
        help="If it is activated, the position and the content of character are not avaible during training.",
    )
    parser.add_argument(
        "--no_con", 
        action="store_true",
        default=False,
        help="If it is activated, the content of character is not avaible during training.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--drop_caption", 
        action="store_true", 
        help="Whether or not to drop captions during training."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='MARIO-10M',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--segmentation_mask_aug", 
        action="store_true", 
        help="Whether to augment the segmentation masks (inspired by https://arxiv.org/abs/2211.13227)."
    ) 
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--image_column", 
        type=str, 
        default="image", 
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0, 
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mask_all_ratio", 
        type=float, 
        default=0.5, 
        help="The training ratio of two branches."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
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
        default=None,
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
        default=None,
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
        "--noise_offset", 
        type=float, 
        default=0, 
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default='/home/cjy/cjy/TextDiffusion/data/laion-ocr-unzip',
        help="The path of dataset."
    )
    parser.add_argument(
        "--train_dataset_index_file", 
        type=str, 
        default='/home/jingyechen/jingyechen/amlt_test/diffusers_combine/examples/text_to_image/train_dataset_index.txt', 
        help="The txt file that provides the index of training samples. The format of each line should be XXXXX_XXXXXXXXX."
    )
    parser.add_argument(
        "--vis_num", 
        type=int, 
        default=16, 
        help="The number of images to be visualized during training."
    )
    parser.add_argument(
        "--vis_interval", 
        type=int, 
        default=500, 
        help="The interval for visualization."
    )


    args = parser.parse_args()
    
    print('***************')
    print(args)
    print('***************')
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    # If passed along, set the training seed now.
    args.seed = random.randint(0, 1000000) if args.seed is None else args.seed
    
    print(f'{colored("[√]", "green")} Arguments are loaded.')
    print(args)
    
    set_seed(args.seed)
    print(f'{colored("[√]", "green")} Seed is set to {args.seed}.')
    
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
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    ) 
    
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
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
        
    from datasets import Dataset
    from datasets import load_dataset
    lines = open(args.train_dataset_index_file).readlines()
    random.shuffle(lines)
    train_dataset = Dataset.from_dict({"image": lines, "text": lines}) 
    dataset = {
        'train': train_dataset,
    }

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names


    dataset_name_mapping = {
        "MARIO-10M": ("image", "text"), 
    }
    
    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            
            caption = caption.strip()
            first, second = caption.split('_')
            try:
                caption = open(f'{args.dataset_path}/{first}/{second}/caption.txt').readlines()[0]
            except:
                caption = 'null'
                print('erorr of caption')
                
            if args.drop_caption and is_train and random.random() < 0.1: 
                caption = '' # drop caption with 10% probability
                        
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    # Please not that Crop is not suitable for this task as texts may be incomplete during cropping
    train_transforms = transforms.Compose( 
        [
            transforms.ToTensor(),
        ]
    )
    
    
    def generate_random_rectangles(image):
        # randomly generate 0~3 masks
        rectangles = []
        box_num = random.randint(0, 3)
        for i in range(box_num):
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            w = random.randint(16, 256)
            h = random.randint(16, 96) 
            angle = random.randint(-45, 45)
            p1 = (x, y)
            p2 = (x + w, y)
            p3 = (x + w, y + h)
            p4 = (x, y + h)
            center = ((x + x + w) / 2, (y + y + h) / 2)
            p1 = rotate_point(p1, center, angle)
            p2 = rotate_point(p2, center, angle)
            p3 = rotate_point(p3, center, angle)
            p4 = rotate_point(p4, center, angle)
            rectangles.append((p1, p2, p3, p4))
        return rectangles


    def rotate_point(point, center, angle):
        # rotation
        angle = math.radians(angle)
        x = point[0] - center[0]
        y = point[1] - center[1]
        x1 = x * math.cos(angle) - y * math.sin(angle)
        y1 = x * math.sin(angle) + y * math.cos(angle)
        x1 += center[0]
        y1 += center[1]
        return int(x1), int(y1)


    def box2point(box):
        # convert string to list
        box = box.split(',')
        box = [int(i)//(512//512) for i in box] 
        points = [(box[0],box[1]),(box[2],box[3]),(box[4],box[5]),(box[6],box[7])]
        return points
    
    
    def get_mask(ocrs):
        # the two branches are trained at a certain ratio
        if random.random() <= args.mask_all_ratio: 
            image_mask = Image.new('L', (512,512), 1)
            return image_mask
        
        image_mask = Image.new('L', (512,512), 0)
        draw_image_mask = ImageDraw.ImageDraw(image_mask)
        for ocr in ocrs:
            ocr = ocr.strip()
            _, box, _ = ocr.split()
            if random.random() < 0.5: # each box is masked with 50% probability
                points = box2point(box)
                draw_image_mask.polygon(points, fill=1) 
        
        blank = Image.new('RGB', (512, 512), (0, 0, 0))
        rectangles = generate_random_rectangles(blank) # get additional masks (can mask non-text areas)
        for rectangle in rectangles:
            draw_image_mask.polygon(rectangle, fill=1)
        
        return image_mask


    def preprocess_train(examples):
        # preprocess the training data
                
        images = []
        segmentation_masks = []
        image_masks = []
        for image in examples[image_column]:
            image = image.strip()
            first, second = image.split('_')
            image_path = f'{args.dataset_path}/{first}/{second}/image.jpg'
            ocrs = open(f'{args.dataset_path}/{first}/{second}/ocr.txt').readlines() 
            
            image = Image.open(image_path).convert("RGB")
            
            image_mask = get_mask(ocrs)
            image_mask_np = np.array(image_mask)
            image_mask_tensor = torch.from_numpy(image_mask_np)
            images.append(image) 
            
            if args.no_pos_con:
                segmentation_mask = np.load(f'{args.dataset_path}/{first}/{second}/charseg.npy') * 0 
            elif args.no_con:
                segmentation_mask = (np.load(f'{args.dataset_path}/{first}/{second}/charseg.npy') > 0).astype(np.float32) 
            else:
                segmentation_mask = np.load(f'{args.dataset_path}/{first}/{second}/charseg.npy') 

            if args.segmentation_mask_aug: # 10% dilate / 10% erode / 10% drop
                random_value = random.random()
                if random_value < 0.6:
                    pass
                elif random_value < 0.7:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=1)
                elif random_value < 0.8:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=1)
                elif random_value < 0.85:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=2)
                elif random_value < 0.9:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=2)
                else:
                    drop_mask = np.random.rand(*segmentation_mask.shape) < 0.1
                    segmentation_mask[drop_mask] = 0 # set character to non-character with 10% probability
            
            segmentation_masks.append(segmentation_mask)
            image_masks.append(image_mask_tensor)
            
        examples["images"] = [train_transforms(image).sub_(0.5).div_(0.5) for image in images] 
        examples["prompts"] = tokenize_captions(examples)
        examples["segmentation_masks"] = segmentation_masks
        examples["image_masks"] = image_masks 
        
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples): 
        images = torch.stack([example["images"] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()
        prompts = torch.stack([example["prompts"] for example in examples])
        image_masks = torch.cat([example["image_masks"].unsqueeze(0) for example in examples],0)
        segmentation_masks = torch.cat([torch.from_numpy(example["segmentation_masks"]).unsqueeze(0).unsqueeze(0) for example in examples], dim=0)
        return {"images": images, "prompts": prompts, "segmentation_masks": segmentation_masks, "image_masks": image_masks}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

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
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    # import segmenter for calculating loss
    from model.text_segmenter.unet import UNet
    segmenter = UNet(4,96, True).cuda() 
    state_dict = torch.load(args.character_aware_loss_ckpt, map_location='cpu') 
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    segmenter.load_state_dict(new_state_dict)
    segmenter.eval()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # Convert images to latent space
                features = vae.encode(batch["images"].to(weight_dtype)).latent_dist.sample()
                features = features * vae.config.scaling_factor

                image_masks = batch["image_masks"]
                
                masked_images = batch["images"] * (1-image_masks).unsqueeze(1) 
                masked_features = vae.encode(masked_images.to(weight_dtype)).latent_dist.sample()
                masked_features = masked_features * vae.config.scaling_factor 
                
                segmentation_masks = batch["segmentation_masks"]
                image_masks_256 = F.interpolate(image_masks.unsqueeze(1), size=(256, 256), mode='nearest')
                segmentation_masks = image_masks_256 * segmentation_masks 
                feature_masks = F.interpolate(image_masks.unsqueeze(1), size=(64, 64), mode='nearest')
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(features)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (features.shape[0], features.shape[1], 1, 1), device=features.device
                    )

                bsz = features.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=features.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(features, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["prompts"])[0]
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon": # √ 
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(features, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
       
                
                if accelerator.is_main_process: 
                    if (step + 1) % args.vis_interval == 0:           
                        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
                        scheduler.set_timesteps(50) 
                        noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda") 
                        input = noise

                        for t in tqdm(scheduler.timesteps):
                            with torch.no_grad():    
                                noisy_residual = unet(input, t, encoder_hidden_states[:args.vis_num], masked_feature=masked_features[:16], feature_mask=feature_masks[:16], segmentation_mask=segmentation_masks[:16]).sample
                                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                                input = prev_noisy_sample

                        # decode
                        input = 1 / vae.config.scaling_factor * input 
                        images = vae.decode(input.half(), return_dict=False)[0] 

                        ## save predicted images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4*width, 4*height))
                        for index, image in enumerate(images.float()):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col*width, row*height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_pred_img.png')

                        ## save segmentation masks
                        width, height = 512, 512
                        new_image = Image.new('L', (4*width, 4*height))
                        for index, image in enumerate(segmentation_masks[:args.vis_num]):
                            segmap_pil = Image.fromarray(((image!=0)*255).squeeze().cpu().numpy().astype("uint8"))
                            row = index // 4
                            col = index % 4
                            new_image.paste(segmap_pil, (col*width, row*height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_segmentation_mask.png')

                        ## save original images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4*width, 4*height))
                        for index, image in enumerate(batch["images"][:args.vis_num]):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            # pred_images.append(image)
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col*width, row*height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_orig_img.png')
                        
                        ## save masked original images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4*width, 4*height))
                        for index, image in enumerate(masked_images[:args.vis_num]):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            # pred_images.append(image)
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col*width, row*height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_masked_orig_img.png')
                        print('inference successfully')

                model_pred = unet(
                    sample=noisy_latents, 
                    timestep=timesteps, 
                    encoder_hidden_states=encoder_hidden_states, 
                    masked_feature=masked_features, 
                    feature_mask=feature_masks, 
                    segmentation_mask=segmentation_masks
                ).sample  
    
                pred_x0 = noise_scheduler.get_x0_from_noise(model_pred, timesteps, noisy_latents)
                resized_charmap = F.interpolate(batch["segmentation_masks"].float(), size=(64, 64), mode="nearest").long()
                
                ce_loss = ce_criterion(segmenter(pred_x0.float()), resized_charmap.squeeze(1))
                mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
                loss = mse_loss + ce_loss * args.character_aware_loss_lambda 
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], 'mse_loss': mse_loss.detach().item(), 'ce_loss': ce_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        ) 
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
