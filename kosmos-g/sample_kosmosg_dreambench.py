import os

import torch
from PIL import Image
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.multimodal.clip_score import CLIPScore as CLIP_TScore
from tqdm import tqdm

from app_model import AppModel
from app_utils import randomize_seed_fn
from eval.clip_score import CLIPIScore as CLIP_IScore
from eval.clip_score import CLIPTScore as CLIP_TScore
from eval.dino_score import DINOScore as DINO_Score
from eval.dreambench_prompts import *
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, files):
        self.args = args
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        object_name, object_id, image_id, prompt = image_path.split('/')[-1].split('.')[0].split('+')
        image = Image.open(image_path).convert('RGB')
        real_image = Image.open(os.path.join(self.args.data_dir, object_name, object_id + '.jpg')).convert('RGB')

        return image, real_image, prompt


def image_collate_fn(batch):
    image = [x[0] for x in batch]
    real_image = [x[1] for x in batch]
    prompt = [x[2] for x in batch]
    return image, real_image, prompt


class DreamBench_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, preprocess_fn):
        self.args = args
        self.preprocess_fn = preprocess_fn
        # Traverse all images in the dataset
        self.image_paths = []
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths) * 25

    def __getitem__(self, index):
        image_path = self.image_paths[index // 25]
        real_image = Image.open(image_path).convert('RGB')
        object_id = image_path.split('/')[-1].split('.')[0]
        object_name = image_path.split('/')[-2]
        if object_name in OBJECT:
            object_class = OBJECT[object_name]
            prompt = OBJECT_PROMPTS[index % 25]
            input_prompt = KOSMOSG_OBJECT_PROMPTS[index % 25]
        else:
            object_class = LIVE_OBJECT[object_name]
            prompt = LIVE_OBJECT_PROMPTS[index % 25]
            input_prompt = KOSMOSG_LIVE_OBJECT_PROMPTS[index % 25]

        prompt = prompt.format(object_class)
        input_prompt = input_prompt.format('<i>' if self.args.drop_object else (object_class + ' <i>'))

        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.preprocess_fn(input_prompt,
                               "" if self.args.negative_prompt else "",
                               real_image, single_batch=False)

        return src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, object_name, object_id, \
            real_image, prompt


def dreambench_collate_fn(batch):
    src_tokens = [x[0] for x in batch]
    gpt_img_src_tokens = torch.cat([x[1] for x in batch])
    img_gpt_input_mask = [x[2] for x in batch]
    negative_tokens = batch[0][3].unsqueeze(0)
    src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=1)
    img_gpt_input_mask = pad_sequence(img_gpt_input_mask, batch_first=True, padding_value=0)
    object_name = [x[4] for x in batch]
    object_id = [x[5] for x in batch]
    real_image = [x[6] for x in batch]
    prompt = [x[7] for x in batch]
    return src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, object_name, object_id, real_image, prompt


def main(cfg):
    cfg.model.pretrained_ckpt_path = "/path/to/trained-ckpt"
    args = OmegaConf.create()
    args.data_dir = "/path/to/dreambench"
    args.batch_size = 5
    args.num_workers = 4
    args.scheduler = "dpms"  # ['ddim', 'pndm', 'dpms']
    args.num_inference_steps = 100
    args.guidance_scale = 7.5
    args.num_images_per_prompt = 4
    args.seed = 0
    args.negative_prompt = False
    args.drop_object = True
    args.output_dir = "/path/to/output-dir/" + cfg.model.pretrained_ckpt_path.split('/')[-2] + '_' \
                      + cfg.model.pretrained_ckpt_path.split('/')[-1].split('.')[0].split('_')[-1] + '_' \
                      + args.scheduler + '_' + str(args.num_inference_steps) + '_' + str(args.guidance_scale) \
                      + '_' + str(args.negative_prompt) + '_' + str(args.drop_object)

    accelerator = Accelerator()
    if accelerator.is_main_process and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dino_score = DINO_Score(model_name_or_path='dino_vits16')
    clip_i_score = CLIP_IScore(model_name_or_path='openai/clip-vit-base-patch32')
    clip_t_score = CLIP_TScore(model_name_or_path='openai/clip-vit-base-patch32')

    dino_score = accelerator.prepare_model(dino_score, evaluation_mode=True)
    clip_i_score = accelerator.prepare_model(clip_i_score, evaluation_mode=True)
    clip_t_score = accelerator.prepare_model(clip_t_score, evaluation_mode=True)

    # stat existing images in output_dir
    image_paths = list()
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    if len(image_paths) >= 3000:
        accelerator.print("Already generated enough images")
        dataset = Image_Dataset(args, image_paths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=args.num_workers,
                                                 shuffle=False, pin_memory=True, drop_last=False,
                                                 persistent_workers=True, collate_fn=image_collate_fn)
        dataloader = accelerator.prepare(dataloader)
        accelerator.print("Number of Images: ", len(dataset))

        for batch in tqdm(dataloader):
            images, real_images, prompts = batch
            dino_score.update(images, real_images)
            clip_i_score.update(images, real_images)
            clip_t_score.update(images, prompts)
        accelerator.print("Computing Scores...")
        accelerator.print("DINO Score: ", dino_score.compute())
        accelerator.print("CLIP Image Score: ", clip_i_score.compute())
        accelerator.print("CLIP Text Score: ", clip_t_score.compute())
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

    dataset = DreamBench_Dataset(args, model.kosmosg_preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False, pin_memory=True,
                                             drop_last=False, persistent_workers=True,
                                             collate_fn=dreambench_collate_fn)
    accelerator.print("Number of Images: ", len(dataset))

    model, dataloader = accelerator.prepare(model, dataloader)

    kwargs = {
        'num_inference_steps': args.num_inference_steps,
        'text_guidance_scale': args.guidance_scale,
        'num_images_per_prompt': args.num_images_per_prompt,
        'lora_scale': 0.0,
    }

    for batch_id, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, object_name, object_id, real_image, prompt = batch

        # generate images
        randomize_seed_fn(args.seed, False)
        images = model.model.sample(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, **kwargs)

        # save image
        for image_id, image in enumerate(images):
            pos = batch_id * accelerator.num_processes * args.batch_size * args.num_images_per_prompt + \
                  image_id * accelerator.num_processes + accelerator.process_index
            name = '+'.join([
                object_name[image_id % args.batch_size],
                object_id[image_id % args.batch_size],
                str(pos),
                prompt[image_id % args.batch_size]
            ])
            images[image_id].save(os.path.join(args.output_dir, "{}.png".format(name)))

        real_image = real_image * args.num_images_per_prompt
        dino_score.update(images, real_image)
        clip_i_score.update(images, real_image)
        clip_t_score.update(images, prompt * args.num_images_per_prompt)

    accelerator.print("Number of Samples: ", (dino_score.n_samples * accelerator.num_processes).item())
    accelerator.print("DINO Score: ", (dino_score.compute()).item())
    accelerator.print("CLIP Image Score: ", (clip_i_score.compute()).item())
    accelerator.print("CLIP Text Score: ", (clip_t_score.compute()).item())


if __name__ == "__main__":
    parser = options.get_training_parser()
    cfg = options.parse_args_and_arch(parser, modify_parser=None)
    cfg = convert_namespace_to_omegaconf(cfg)
    main(cfg)
