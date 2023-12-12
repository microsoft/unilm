# ------------------------------------------
# TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
# Paper Link: https://arxiv.org/abs/2311.16465
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser-2
# Copyright (c) Microsoft Corporation.
# ------------------------------------------

import json
from PIL import Image
import random
import string
from tqdm import tqdm

import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

import argparse
import logging
import math
import os
import random
from pathlib import Path
from PIL import Image

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "MARIO-10M": ("image", "text"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_pertubation", type=float, default=0, help="The scale of input pretubation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vis_num", 
        type=int, 
        default=16, 
        help="The number of images to be visualized during training."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--index_file_path", 
        type=str, 
        default='/home/jingyechen/jingyechen/amlt_test/diffusers_combine/examples/text_to_image/train_dataset_index.txt', 
        help="The txt file that provides the index of training samples. The format of each line should be XXXXX_XXXXXXXXX."
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default='/path/to/laion-ocr-select',
        required=True,
        help="the root of the dataset, please follow the code in textdiffuser-1"
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
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
        default=10,
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--max_length", 
        default=77,
        type=int, 
        help="Maximum length of the prompt. Can enlarge this value to adapt longer coord representation."
    )
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
        help="The way to represent coordinates. Can use one point or two points"
    )
    parser.add_argument(
        "--vis_interval", 
        type=int, 
        default=1000, 
        help="The interval for visualization."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args



#### check whether two boxes can be merged in the x-axis
def check_merge(box1, box2):

    x_center1, y_center1, x_min1, y_min1, x_max1, y_max1, pred1 = box1
    x_center2, y_center2, x_min2, y_min2, x_max2, y_max2, pred2 = box2

    if y_center1 >= y_min2 and y_center1 <= y_max2:
        if y_center2 >= y_min1 and y_center2 <= y_max1:
            pass
        else:
            return False
    else:
        return False

    distance1 = x_max2 - x_min1
    distance2 = (x_max2 - x_min2) + (x_max1 - x_min1)

    if distance2 / distance1 >= 0.8:
        if x_min1 < x_min2:
            pred = pred1 + ' ' + pred2
        else:
            pred = pred2 + ' ' + pred1

        x_min = min(x_min1, x_min2)
        y_min = min(y_min1, y_min2)
        x_max = max(x_max1, x_max2)
        y_max = max(y_max1, y_max2)

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        return [x_center, y_center, x_min, y_min, x_max, y_max, pred]

    else:
        return False


#### merge boxes for training at line-level instead of word-level
def merge_boxes(boxes):
    results = []
    while True:
        if len(boxes) == 0:
            break 
        
        flag = False
        sample = boxes[0] 
        boxes.remove(sample) 
        for item in boxes:
            result = check_merge(sample, item)
            if result:
                boxes.remove(item)
                boxes.append(result) 
                boxes = sorted(boxes, key=lambda x: x[0]) 
                flag = True
                break
            else:
                pass

        if flag is False:
            results.append(sample)

    return results


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

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    #### additional tokens are introduced, including coordinate tokens and character tokens
    print('[Size of the original tokenizer] ', len(tokenizer))
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) # character-level embedding

    print('[Size of the modified tokenizer] ', len(tokenizer))

    if args.max_length == 77:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
    else:
        #### enlarge the context length of text encoder. empirically, enlarging the context length can proceed longer sequence. However, we observe that it will be hard to render general objects
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, max_position_embeddings=args.max_length, ignore_mismatched_sizes=True
        )
    text_encoder.resize_token_embeddings(len(tokenizer))

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder
    #### this script is provided for full-parameter fine-tuning, so the unet is trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(True) #### the text_encoder should be trainable to learn the newly-added tokens

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

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

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

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
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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

    #### train the u-net and text encoder
    #### the lr for training u-net is set to a lower value (1e-5) following textdiffuser-1
    optimizer = optimizer_cls(
        [
            {'params': text_encoder.parameters(), 'lr': 1e-5},
            {'params': unet.parameters(), 'lr': 1e-5},
        ], 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon
    )

    #### args.train_dataset_index_file contains multiple lines, each of which should follow the format 00123_0012304567 ...
    from datasets import Dataset
    lines = open(args.index_file_path).readlines()
    random.shuffle(lines)
    train_dataset = Dataset.from_dict({"image": lines, "text": lines}) 
    dataset = {
        'train': train_dataset,
    }

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
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

    #### many augmentations can not be used in the text rendering task 
    train_transforms = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )

    #### process the training data
    def preprocess_train(examples):            
        images = []
        prompts_train = []
        prompts_cond = []
        prompts_nocond = []
        for image in examples[image_column]:
            image = image.strip()
            first, second = image.split('_')
            
            #### get image
            image_path = f'{args.dataset_path}/{first}/{second}/image.jpg'
            image = Image.open(image_path).convert("RGB")
            images.append(image)
           
            #### get caption
            try: #### note that few cases do not contain valid captions
                caption = open(f'{args.dataset_path}/{first}/{second}/caption.txt').readlines()[0]
            except:
                caption = 'null'
                print('erorr of caption')
            
            #### get ocr
            #### since the original ocr annotations are word-level, we need to merge some boxes to construct line-level ocr
            ocrs = open(f'{args.dataset_path}/{first}/{second}/ocr.txt').readlines()

            ocrs_temp = []
            for line in ocrs:
                line = line.strip()
                pred, box, prob = line.split()
                items = box.split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = int(items[0]), int(items[1]), int(items[2]), int(items[3]), int(items[4]), int(items[5]), int(items[6]), int(items[7])
                x_min = min(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                x_max = max(x1, x2, x3, x4)
                y_max = max(y1, y2, y3, y4)
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                ocrs_temp.append([x_center, y_center, x_min, y_min, x_max, y_max, pred])
            ocrs_temp = sorted(ocrs_temp, key=lambda x: x[0])
            ocrs_temp = merge_boxes(ocrs_temp)
            ocrs_temp = sorted(ocrs_temp, key=lambda x: x[1])

            random.shuffle(ocrs_temp) #### augment the ocr sequence for robust training

            ocr_ids = [] #### concat with the prompt tokens
            for line in ocrs_temp:

                x_center, y_center, x_min, y_min, x_max, y_max, pred = line
                
                # choose coord mode
                if args.coord_mode == 'lt':
                    x_left = x_min
                    y_top = y_min
                    x_left = x_left // (512 // args.granularity)
                    y_top = y_top // (512 // args.granularity)
                    x_left = np.clip(x_left, 0, args.granularity)
                    y_top = np.clip(y_top, 0, args.granularity)
                    ocr_ids.extend(['l'+str(x_left), 't'+str(y_top)])

                elif args.coord_mode == 'center':
                    x_center = x_center // (512 // args.granularity)
                    y_center = y_center // (512 // args.granularity)
                    x_center = np.clip(x_center, 0, args.granularity)
                    y_center = np.clip(y_center, 0, args.granularity)
                    ocr_ids.extend(['l'+str(x_center), 't'+str(y_center)])
                    
                elif args.coord_mode == 'ltrb':
                    x_left = x_min
                    y_top = y_min
                    x_right = x_max
                    y_bottom = y_max
                    x_left = x_left // (512 // args.granularity)
                    y_top = y_top // (512 // args.granularity)
                    x_right = x_right // (512 // args.granularity)
                    y_bottom = y_bottom // (512 // args.granularity)
                    x_left = np.clip(x_left, 0, args.granularity)
                    y_top = np.clip(y_top, 0, args.granularity)
                    x_right = np.clip(x_right, 0, args.granularity)
                    y_bottom = np.clip(y_bottom, 0, args.granularity)
                    ocr_ids.extend(['l'+str(x_left), 't'+str(y_top), 'r'+str(x_right), 'b'+str(y_bottom)])

                char_list = list(pred)
                char_list = [f'[{i}]' for i in char_list]
                ocr_ids.extend(char_list)
                ocr_ids.append(tokenizer.eos_token_id)
            ocr_ids.append(tokenizer.eos_token_id) 

            ocr_ids = tokenizer.encode(ocr_ids)

            caption_ids = tokenizer(
                caption, truncation=True, return_tensors="pt"
            ).input_ids[0].tolist() 

            prompt = caption_ids + ocr_ids
            prompt = prompt[:args.max_length]
            while len(prompt) < args.max_length: 
                prompt.append(tokenizer.pad_token_id) 

            prompts_cond.append(prompt)
            prompts_nocond.append([tokenizer.pad_token_id]*args.max_length)

            #### classifier-free guidance
            if random.random() < 0.1:
                prompts_train.append([tokenizer.pad_token_id]*args.max_length)
            else:
                prompts_train.append(prompt)

        examples["images"] = [train_transforms(image).sub_(0.5).div_(0.5) for image in images]  
        examples["prompts_train"] = prompts_train 
        examples["prompts_cond"] = prompts_cond
        examples["prompts_nocond"] = prompts_nocond

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)


    def collate_fn(examples): 
        images = torch.stack([example["images"] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()

        prompts_train = torch.Tensor([example["prompts_train"] for example in examples]).long()
        prompts_cond = torch.Tensor([example["prompts_cond"] for example in examples]).long()
        prompts_nocond = torch.Tensor([example["prompts_nocond"] for example in examples]).long()
        return {"images": images, "prompts_train": prompts_train, "prompts_cond": prompts_cond, "prompts_nocond": prompts_nocond}

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
    #### please note that "text_encoder" should be added for training
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
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
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
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
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

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
            try:
                accelerator.load_state(args.resume_from_checkpoint)
            except:
                accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        text_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["images"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_pertubation:
                    new_noise = noise + args.input_pertubation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_pertubation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompts_train"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")




                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


                #### visualization during training
                if True:
                # if accelerator.is_main_process: 
                    
                    cfg = 7
                    if (step + 0) % args.vis_interval == 0:           
                        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
                        scheduler.set_timesteps(50) 
                        noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda") 
                        input = noise

                        encoder_hidden_states_cond = text_encoder(batch["prompts_cond"])[0] 
                        encoder_hidden_states_nocond = text_encoder(batch["prompts_nocond"])[0] 

                        texts = batch["prompts_cond"]
                        
                        f = open(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_prompt_{args.local_rank}.txt', 'w+')
                        for text in texts:
                            # also need to call a function to map back
                            sentence = tokenizer.decode(text)
                            f.write(sentence + '\n')
                        f.close()

                        for t in tqdm(scheduler.timesteps):
                            with torch.no_grad():  # classifier free guidance

                                noise_pred_cond = unet(sample=input.half(), timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:args.vis_num]).sample # b, 4, 64, 64
                                noise_pred_uncond = unet(sample=input.half(), timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:args.vis_num]).sample # b, 4, 64, 64
                                noisy_residual = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
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
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_pred_img_cfg{cfg}_{args.local_rank}.jpg')
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
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_orig_img_{args.local_rank}.jpg')
                       
                        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
                        scheduler.set_timesteps(50) 
                        noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda") 
                        input = noise

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

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
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
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
