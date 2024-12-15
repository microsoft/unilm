import argparse
import functools
import logging
import math
import os
from datetime import timedelta

import datasets
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from datasets import load_dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import diffusers
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler

from models import All_models, DiT, Transformer, EMAModel
from timm.models import create_model
from utils import center_crop_arr, safe_blob_write, load_vae
from schedule.ddpm import DDPMScheduler

import wandb

logger = get_logger(__name__, log_level="INFO")


def parse_args():  
    parser = argparse.ArgumentParser(description="Simple example of a training script.")  
  
    # 基本参数  
    parser.add_argument("--seed", type=int, default=0, help="A seed to use for the random number generator. Can be negative to not set a seed.")  
    parser.add_argument("--output_dir", type=str, default="results", help="The output directory where the model predictions and checkpoints will be written.")  
    parser.add_argument("--cache_dir", type=str, default="/mnt/msranlp/yutao/cache", help="The directory where the downloaded models and datasets will be stored.")  
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")  
  
    # 数据集参数  
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the Dataset (from the HuggingFace hub) to train on.")  
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The config of the Dataset, leave as None if there's only one config.")  
    parser.add_argument("--train_data_dir", type=str, default="/tmp/ILSVRC/Data/CLS-LOC/train", help="A folder containing the training data.")  
      
    # 模型参数  
    parser.add_argument("--model", type=str, default="Transformer-L", help="The config of the UNet model to train.")  
    parser.add_argument("--vae", type=str, default=None, help="Path to pre-trained VAE model.")  
    parser.add_argument("--image_size", type=int, default=256, help="The image_size for input images.")  
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes for the model.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")  
  
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader.")  
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")  
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="The number of subprocesses to use for data loading.")  
      
    # 优化器参数  
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")  
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use.")  
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler.")  
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")  
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="The beta2 parameter for the Adam optimizer.")  
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay magnitude for the Adam optimizer.")  
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")  
      
    # EMA参数  
    parser.add_argument("--use_ema", action="store_true", help="Whether to use Exponential Moving Average for the final model weights.")  
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")  
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")  
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")  
      
    # 日志参数  
    parser.add_argument("--logger", type=str, default=None, help="The logger type to use.")  
    parser.add_argument("--logging_dir", type=str, default="logs", help="The directory to store logs.")  
    parser.add_argument("--wandb_project", type=str, default=None, help="The wandb project name.")  
    parser.add_argument("--wandb_entity", type=str, default=None, help="The wandb entity (username or team).")  
    parser.add_argument("--log_every", type=int, default=100, help="Log every X steps.")  
      
    # 分布式训练参数  
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")  

    # DDPM参数  
    parser.add_argument("--prediction_type", type=str, default="epsilon", help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")  
    parser.add_argument("--ddpm_num_steps", type=int, default=1000, help="The number of steps to use for DDPM.")  
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000, help="The number of inference steps to use for DDPM.")
    parser.add_argument("--ddpm_beta_schedule", type=str, default="cosine", help="The beta schedule to use for DDPM.") 
    parser.add_argument("--ddpm_batch_mul", type=int, default=4, help="The batch multiplier to use for DDPM.")  
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save a checkpoint of the training state every X updates.")  
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a previous checkpoint.")  
      
    args = parser.parse_args()
      
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))  
    if env_local_rank != -1 and env_local_rank != args.local_rank:  
        args.local_rank = env_local_rank  
    if args.dataset_name is None and args.train_data_dir is None:  
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")  
      
    return args  


def main(args):
    set_seed(args.seed)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    vae, input_size, latent_size, flatten_input = load_vae(args.vae, args.image_size)
        
    model = All_models[args.model](
        input_size=input_size,
        in_channels=latent_size,
        num_classes=args.num_classes,
        flatten_input=flatten_input,
        drop=args.dropout,
    )
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            min_decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
        )
    # Initialize the scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high image_size or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
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
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    logger.info(args)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.wandb_project is not None:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)  

    logger.info(model)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    augmentations = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}
        dataset.set_transform(transform_images)
    else:
        dataset = ImageFolder(args.train_data_dir, transform=augmentations)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps),
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # vae = accelerator.prepare_model(vae, evaluation_mode=True, device_placement=True)
    vae.to(accelerator.device)
    vae.eval()
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    max_train_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    running_loss = 0
    first_epoch = 0
    scaling_factor = None
    bias_factor = None
    # Potentially load in the weights and states from a previous save
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and os.path.exists(os.path.join(args.output_dir, "latest")):
        with open(os.path.join(args.output_dir, "latest"), "r") as f:
            checkpoint_path = f.read().strip()
        
    if checkpoint_path is not None:
        accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        other_state = torch.load(os.path.join(checkpoint_path, "other_state.pth"), map_location="cpu")
        global_step = other_state["steps"]
        scaling_factor = other_state["scaling_factor"]
        bias_factor = other_state["bias_factor"]
        if args.use_ema:
            ema_model.load_state_dict(other_state["ema"])
            logger.info("EMA model loaded successfully")
        first_epoch = global_step * args.gradient_accumulation_steps // len(train_dataloader)
        resume_step = global_step * args.gradient_accumulation_steps % len(train_dataloader)

    # Train!
    # snr = compute_snr(noise_scheduler, torch.arange(args.ddpm_num_steps, device=accelerator.device))
    # sample_weight = (
    #     torch.stack([snr, 5 * torch.ones(args.ddpm_num_steps, device=accelerator.device)], dim=1).min(dim=1)[0] / snr
    # )
    sample_weight = torch.ones(args.ddpm_num_steps, device=accelerator.device)
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        for step, (clean_images, label) in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.checkpoint and epoch == first_epoch:
                if step < resume_step:
                    continue
                
            with torch.no_grad():
                vae_latent = vae.encode(clean_images)
                clean_images = vae_latent.sample()
                mode_images = vae_latent.mode()
                if scaling_factor is None:
                    scaling_factor = 1. / clean_images.flatten().std()
                    bias_factor = -clean_images.flatten().mean()
                    dist.all_reduce(scaling_factor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(bias_factor, op=dist.ReduceOp.SUM)
                    scaling_factor = scaling_factor.item() / dist.get_world_size()
                    bias_factor = bias_factor.item() / dist.get_world_size()
                    logger.info(f"Scaling factor: {scaling_factor}, Bias factor: {bias_factor}")
                clean_images = (clean_images + bias_factor) * scaling_factor
                mode_images = (mode_images + bias_factor) * scaling_factor

            with accelerator.accumulate(model):
                bsz, latent_size, h, w = clean_images.shape
                if isinstance(model.module, Transformer):
                    noise = torch.randn((bsz * args.ddpm_batch_mul * h * w, latent_size), device=clean_images.device, dtype=clean_images.dtype)
                    timesteps = torch.multinomial(sample_weight, bsz * args.ddpm_batch_mul * h * w, replacement=True)
                    clean_images_repeated = clean_images.repeat_interleave(args.ddpm_batch_mul, dim=0).permute(0, 2, 3, 1).reshape(-1, clean_images.shape[1])
                    noisy_images = noise_scheduler.add_noise(clean_images_repeated, noise, timesteps)
                    velocity = noise_scheduler.get_velocity(clean_images_repeated, noise, timesteps)
                    noisy_images, noise, velocity = [x.reshape(bsz * args.ddpm_batch_mul, h, w, latent_size).permute(0, 3, 1, 2) for x in [noisy_images, noise, velocity]]
                    timesteps = timesteps.reshape(bsz * args.ddpm_batch_mul, h * w)
                    model_output = model(noisy_images.to(dtype), timesteps.to(dtype), x_start=clean_images.to(dtype), y=label, batch_mul=args.ddpm_batch_mul)
                elif isinstance(model.module, DiT):
                    noise = torch.randn_like(clean_images)
                    timesteps = torch.multinomial(sample_weight, bsz, replacement=True)
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    velocity = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                    model_output = model(noisy_images.to(dtype), timesteps.to(dtype), y=label)
                else:
                    raise NotImplementedError()
                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())
                elif args.prediction_type == "v_prediction":
                    loss = F.mse_loss(model_output.float(), velocity.float())
                else:
                    raise NotImplementedError()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()                 

            running_loss += loss.item()
            if accelerator.sync_gradients:
                global_step += 1
                if args.use_ema:
                    ema_model.step(model.parameters())
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / args.log_every / args.gradient_accumulation_steps
                    running_loss = 0
                    logs = {"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "gnorm": gnorm.item(), "batch size": total_batch_size, "epoch": epoch}
                    if args.use_ema:
                        logs["ema_decay"] = ema_model.cur_decay_value
                    logger.info(logs)
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process and args.wandb_project is not None:  
                        wandb.log(logs, step=global_step)
                
                if global_step % args.checkpointing_steps == 0:
                    def save_checkpoint(path):
                        accelerator.save_state(path)
                        if accelerator.is_main_process:
                            other_state = {
                                "scaling_factor": scaling_factor,
                                "bias_factor": bias_factor,
                                "steps": global_step,
                                "ema": ema_model.state_dict() if args.use_ema else None,
                            }
                            torch.save(other_state, os.path.join(path, "other_state.pth"))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_checkpoint(os.path.join(save_path))
                    if accelerator.is_main_process:
                        safe_blob_write(os.path.join(args.output_dir, "latest"), save_path)
                    logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)