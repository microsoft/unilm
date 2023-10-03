import base64
import io
import multiprocessing
import os
import random
from argparse import ArgumentParser
from multiprocessing import Process

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label, find_objects, grey_dilation
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, \
    CLIPSegProcessor, CLIPSegForImageSegmentation

Image.MAX_IMAGE_PIXELS = 1000000000

INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "### Input:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
Extract all objects mentioned in the caption and separate them using commas. Exclude background elements (site, location, environment) and only include foreground objects. Ensure that only nouns are included and exclude adjectives entirely.

{input_key}
{input}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
)


@torch.no_grad()
def save_tsv(args, shard_id, shard, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    model_dtype = torch.float16
    # blip2
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=model_dtype)
    blip2_model.eval().to(device)

    # mpt
    mpt_config = AutoConfig.from_pretrained('mosaicml/mpt-7b-instruct', trust_remote_code=True)
    mpt_config.init_device = device
    mpt_config.attn_config['attn_impl'] = args.attn_impl

    mpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    mpt_tokenizer.pad_token = mpt_tokenizer.eos_token
    mpt_tokenizer.padding_side = 'left'
    mpt_model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-7b-instruct', config=mpt_config,
                                                     torch_dtype=model_dtype, trust_remote_code=True)
    mpt_model.eval()

    mpt_generate_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'use_cache': args.use_cache,
        'do_sample': False if args.temperature == 0 else args.do_sample,
        'eos_token_id': mpt_tokenizer.eos_token_id,
        'pad_token_id': mpt_tokenizer.pad_token_id,
    }

    # clipseg
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", torch_dtype=model_dtype)
    clipseg_model.eval().to(device)

    cnt = 0

    for image in tqdm(shard):
        if image is None:
            continue
        if cnt % 1000 == 0:
            # close previous file if any
            if cnt > 0:
                f.close()
            f = open(os.path.join(args.output_dir, f"cnt_{args.machine_id}_{shard_id}_{cnt // 1000}.tsv"), "w",
                     encoding='utf-8')
        cnt += 1

        blip2_input = blip2_processor(images=image, return_tensors="pt").to(device, model_dtype)

        blip2_gen = blip2_model.generate(**blip2_input)
        caption = blip2_processor.batch_decode(blip2_gen, skip_special_tokens=True)[0] \
            .replace('\t', '').replace('\n', '').strip()

        # tag extraction
        prompt = PROMPT_FOR_GENERATION_FORMAT.format(input=caption)

        # Run HF generate
        mpt_input = mpt_tokenizer(prompt, return_tensors='pt', padding=True)
        for key, value in mpt_input.items():
            mpt_input[key] = value.to(device)
        mpt_gen = mpt_model.generate(
            input_ids=mpt_input['input_ids'],
            attention_mask=mpt_input['attention_mask'],
            **mpt_generate_kwargs,
        )
        tags = mpt_tokenizer.batch_decode(mpt_gen, skip_special_tokens=True)[0][len(prompt):]

        if '#' in tags:
            continue
        tags = tags.split(",")

        tags = [tag.replace('\t', '').replace('\n', '').strip() for tag in tags]
        tags = [tag for tag in tags if len(tag) > 0 and tag in caption]

        if len(tags) == 0:
            continue

        clipseg_input = clipseg_processor(text=tags, images=[image] * len(tags), padding=True, return_tensors="pt")
        for key, value in clipseg_input.items():
            clipseg_input[key] = value.to(device)
            if value.dtype == torch.float32:
                clipseg_input[key] = value.to(device, model_dtype)

        # predict
        clipseg_gen = clipseg_model(**clipseg_input).logits

        if len(tags) == 1:
            clipseg_gen = clipseg_gen.unsqueeze(0)

        image_size = image.height

        # interpolate to original size
        clipseg_gen = F.interpolate(clipseg_gen.unsqueeze(1), size=image_size, mode='bilinear')
        masks = torch.sigmoid(clipseg_gen).squeeze(1)
        masks = masks.cpu().numpy()

        sub_images = []
        tags_to_keep = []

        # save the masked image
        for mask_id, mask in enumerate(masks):
            image_array = np.array(image)
            thresholded_mask = mask > args.threshold

            if thresholded_mask.max() == 0:
                continue

            thresholded_mask = grey_dilation(thresholded_mask, size=(image_size // 100, image_size // 100))
            labeled_matrix, num_features = label(thresholded_mask)
            regions = find_objects(labeled_matrix)
            sizes = [np.sum(thresholded_mask[region]) for region in regions]
            max_index = np.argmax(sizes)
            max_region = regions[max_index]
            thresholded_mask[labeled_matrix != (max_index + 1)] = False

            tags_to_keep.append(tags[mask_id])

            # Determine the dimensions of the region
            y_start, y_stop = max_region[0].start, max_region[0].stop
            x_start, x_stop = max_region[1].start, max_region[1].stop
            height = y_stop - y_start
            width = x_stop - x_start

            # Calculate the desired side length for a square region
            side_length = max(height, width)

            # Calculate the center of the region
            center_y = (y_start + y_stop) // 2
            center_x = (x_start + x_stop) // 2

            # Calculate the new boundaries for the region
            new_y_start = center_y - (side_length // 2)
            new_y_stop = new_y_start + side_length
            new_x_start = center_x - (side_length // 2)
            new_x_stop = new_x_start + side_length

            # Adjust the boundaries if they exceed the image boundaries
            if new_y_start < 0:
                new_y_start = 0
                new_y_stop = side_length
            elif new_y_stop > image_array.shape[0]:
                new_y_start = image_array.shape[0] - side_length
                new_y_stop = image_array.shape[0]

            if new_x_start < 0:
                new_x_start = 0
                new_x_stop = side_length
            elif new_x_stop > image_array.shape[1]:
                new_x_start = image_array.shape[1] - side_length
                new_x_stop = image_array.shape[1]

            # Create a new mask with the adjusted boundaries
            object_image = image_array[new_y_start:new_y_stop, new_x_start:new_x_stop]
            max_region_mask = thresholded_mask[new_y_start:new_y_stop, new_x_start:new_x_stop]

            masked_image = object_image.copy()
            masked_image[~max_region_mask] = 255

            object_image = Image.fromarray(object_image).resize((512, 512))
            masked_image = Image.fromarray(masked_image).resize((512, 512))
            sub_images.extend([object_image, masked_image])

        if len(sub_images) == 0:
            continue

        image = image.resize((512, 512))

        # encode image using base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        for j, im in enumerate(sub_images):
            buffer = io.BytesIO()
            im.save(buffer, format='PNG')
            sub_images[j] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # write to tsv file
        f.write('\t'.join([
            caption,
            ','.join(tags_to_keep),
            image,
            *sub_images
        ]) + '\n')


class OpenImageDataset(Dataset):
    def __init__(self, url_data):
        self.data = url_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            items = self.data[idx].split(',')
            image = Image.open(requests.get(items[2], stream=True).raw).convert('RGB')
            # caption
            width, height = image.size
            shortest_side = min(width, height)
            left = (width - shortest_side) // 2
            top = (height - shortest_side) // 2
            right = left + shortest_side
            bottom = top + shortest_side
            image = image.crop((left, top, right, bottom))
            return image
        except:
            return None


def collate_fn(batch):
    return batch[0] if batch is not None else None


def main():
    """Parse commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/path/to/image_ids_and_rotation.csv')
    parser.add_argument('--output-dir', type=str, default='/path/to/output-dir/')
    parser.add_argument('--num-process', type=int, default=8)
    parser.add_argument('--cuda-device', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--num-machine', type=int, default=1)
    parser.add_argument('--machine-id', type=int, default=0)

    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=10)

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--do-sample', type=bool, default=True)
    parser.add_argument('--use-cache', type=bool, default=True)
    parser.add_argument('--trust-remote-code', type=bool, default=True)
    parser.add_argument('--attn-impl', type=str, default='torch')
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    with open(args.data_dir, 'r', encoding='utf8') as f:
        url_data = f.read().strip().split('\n')

    # split into 8 machine, and pick the part of machine_id
    url_data = url_data[args.machine_id::args.num_machine]

    # split url data into shards
    url_data = [url_data[i::args.num_process] for i in range(args.num_process)]

    dataloaders = [
        DataLoader(
            OpenImageDataset(url_data[i]),
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            prefetch_factor=4,
            collate_fn=collate_fn
        )
        for i in range(args.num_process)
    ]

    multiprocessing.set_start_method('spawn')
    processes = []

    for shard_id, shard in enumerate(dataloaders):
        p = Process(
            target=save_tsv,
            args=(
                args,
                shard_id,
                shard,
                torch.device('cuda:{}'.format(args.cuda_device[shard_id % len(args.cuda_device)]))
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('Done!')


if __name__ == '__main__':
    main()
