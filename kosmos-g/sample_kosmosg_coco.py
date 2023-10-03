import json
import os
import random

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
from tqdm import tqdm

from app_model import AppModel
from app_utils import randomize_seed_fn
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


class COCO_Dataset_Image(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        real_image = np.array(Image.open(filename).convert('RGB'))
        real_image = torch.tensor(real_image)
        real_image = real_image.permute(2, 0, 1) / 255.0
        real_image = F.resize(real_image, 256)
        real_image = F.center_crop(real_image, (256, 256))
        return real_image


class COCO_Dataset_Caption(torch.utils.data.Dataset):
    def __init__(self, args, preprocess_fn):
        self.args = args
        self.preprocess_fn = preprocess_fn
        # get text prompts
        with open(os.path.join(args.data_dir, 'annotations', 'captions_val2014.json'), 'r') as f:
            self.coco = json.load(f)
        self.files = self.coco['annotations']
        # random sampled 30K images from COCO
        random.seed(args.seed)
        self.files = random.sample(self.files, 30000)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        prompt = self.files[index]['caption']

        src_tokens, _, img_gpt_input_mask, negative_tokens = \
            self.preprocess_fn(prompt,
                               "" if self.args.negative_prompt else "",
                               None, single_batch=False)

        return src_tokens, img_gpt_input_mask, negative_tokens


def collate_fn(batch):
    src_tokens = [x[0] for x in batch]
    img_gpt_input_mask = [x[1] for x in batch]
    negative_tokens = batch[0][2].unsqueeze(0)
    src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=1)
    img_gpt_input_mask = pad_sequence(img_gpt_input_mask, batch_first=True, padding_value=0)

    return src_tokens, img_gpt_input_mask, negative_tokens


def main(cfg):
    cfg.model.pretrained_ckpt_path = "/path/to/trained-ckpt"
    args = OmegaConf.create()
    args.data_dir = "/path/to/coco"
    args.batch_size = 16
    args.num_workers = 4
    args.scheduler = "ddim"  # ['ddim', 'pndm', 'dpms']
    args.num_inference_steps = 250
    args.guidance_scale = 3.0
    args.num_images_per_prompt = 1
    args.seed = 0
    args.negative_prompt = False
    args.override = False
    args.output_dir = "/path/to/output-dir/" + cfg.model.pretrained_ckpt_path.split('/')[-2] + '_' + \
                      cfg.model.pretrained_ckpt_path.split('/')[-1].split('.')[0].split('_')[-1] + '_' + args.scheduler \
                      + '_' + str(args.num_inference_steps) + '_' + str(args.negative_prompt)

    accelerator = Accelerator()
    if accelerator.is_main_process and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fid = FrechetInceptionDistance(normalize=True)
    fid = accelerator.prepare_model(fid, evaluation_mode=True)
    with open(os.path.join(args.data_dir, 'annotations', 'captions_val2014.json'), 'r') as f:
        files = json.load(f)['images']
    files = [os.path.join(args.data_dir, 'val2014', file['file_name']) for file in files]
    image_dataset = COCO_Dataset_Image(files)
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, num_workers=args.num_workers,
                                                   shuffle=False, pin_memory=True, drop_last=False,
                                                   persistent_workers=True)
    image_dataloader = accelerator.prepare(image_dataloader)
    accelerator.print("Number of real images: ", len(image_dataset))

    for batch in tqdm(image_dataloader):
        fid.update(batch, real=True)

    # stat existing images in output_dir
    image_paths = list()
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    if len(image_paths) >= 30000 and not args.override:
        accelerator.print("Already generated enough images")
        image_dataset = COCO_Dataset_Image(image_paths)
        image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=128, num_workers=args.num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=False,
                                                       persistent_workers=True)
        image_dataloader = accelerator.prepare(image_dataloader)
        accelerator.print("Number of fake images: ", len(image_dataset))

        for batch in tqdm(image_dataloader):
            fid.update(batch, real=False)
        accelerator.print("FID: ", fid.compute())
        return
    else:
        # clear all existing images
        if accelerator.is_main_process:
            for root, dirs, files in os.walk(args.output_dir):
                for file in files:
                    if file.endswith(".png"):
                        os.remove(os.path.join(root, file))

    model = AppModel(cfg)
    model.set_ckpt_scheduler_fn(cfg.model.pretrained_ckpt_path, args.scheduler)

    caption_dataset = COCO_Dataset_Caption(args, model.kosmosg_preprocess)
    caption_dataloader = torch.utils.data.DataLoader(caption_dataset, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, shuffle=False, pin_memory=True,
                                                     drop_last=False, persistent_workers=True, collate_fn=collate_fn)
    accelerator.print("Number of prompts: ", len(caption_dataset))

    model, caption_dataloader = accelerator.prepare(model, caption_dataloader)

    kwargs = {
        'num_inference_steps': args.num_inference_steps,
        'text_guidance_scale': args.guidance_scale,
        'num_images_per_prompt': args.num_images_per_prompt,
        'lora_scale': 0.0,
        'output_type': 'numpy'
    }

    for batch_id, batch in tqdm(enumerate(caption_dataloader), total=len(caption_dataloader)):
        src_tokens, img_gpt_input_mask, negative_tokens = batch
        # generate images
        randomize_seed_fn(args.seed, False)
        images = model.model.sample(src_tokens, None, img_gpt_input_mask, negative_tokens, **kwargs)

        # save image
        for image_id, image in enumerate(images):
            pos = batch_id * accelerator.num_processes * args.batch_size * args.num_images_per_prompt + \
                  image_id * accelerator.num_processes + accelerator.process_index
            model.model.vae.numpy_to_pil(image)[0].save(os.path.join(args.output_dir, "{:05d}.png".format(pos)))

        images = np.stack(images, axis=0)
        images = torch.tensor(images).to(accelerator.device)
        images = images.permute(0, 3, 1, 2)
        fid.update(images, real=False)

    accelerator.print("Number of Real Images: ", (fid.real_features_num_samples * accelerator.num_processes).item())
    accelerator.print("Number of Fake Images: ", (fid.real_features_num_samples * accelerator.num_processes).item())
    accelerator.print("FID: ", fid.compute())


if __name__ == "__main__":
    parser = options.get_training_parser()
    cfg = options.parse_args_and_arch(parser, modify_parser=None)
    cfg = convert_namespace_to_omegaconf(cfg)
    main(cfg)
