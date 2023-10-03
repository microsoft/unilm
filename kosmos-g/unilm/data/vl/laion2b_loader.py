try:
    from fairseq.data.encoders.gpt2_bpe import GPT2BPE
except:
    print("GPT2BPE not found, please install fairseq first if you want to use GPT2BPE")
import base64
import io
import os

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import numpy as np
import torch
from PIL import Image
from tiktoken.core import Encoding
from torchvision.transforms import CenterCrop, Compose, Resize
from infinibatch import iterators
from unilm.data.vl.vl_base_loader import VLBaseLoader
from transformers import CLIPTokenizer

ALT_KEY = 'MMAltTextWords'
CAPTION_KEY = 'MMCaptionWords'
CONTENT_KEY = 'Content'
IMAGE_KEY = 'MMImage'
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"


class NumpyNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor).
        Returns:
        """
        image = np.array(img).transpose(2, 0, 1)  # B, H, W, C  -> B, C, H, W
        image = image / 255.0
        image -= np.array(self.mean).reshape(-1, 1, 1)
        image /= np.array(self.std).reshape(-1, 1, 1)
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Laion2BLoader(VLBaseLoader):
    def _setup(self):
        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.dictionary.add_symbol(BOI_SYMBOL)
        self.dictionary.add_symbol(EOI_SYMBOL)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer",
                                                            torch_dtype=torch.float16, revision="fp16")

    def _build_filter(self):
        def width_height_filter(item):
            # judge item[3] and item[4] is interger
            if item[3].isdigit() and item[4].isdigit():
                return not self.args.align and (int(item[3]) < 384 or int(item[4]) < 384)
            return True

        def length_filter(item):
            if self.args.align and (len(self.clip_tokenizer.tokenize(item[1])) > 75 or len(
                    self.text_transform(item[1])) > 85):
                return True
            return False

        return [width_height_filter, length_filter]

    def _build_image_transform(self):
        preprocess_image = {
            'gpt': Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]),
            'diff': Compose([
                Resize(512),
                CenterCrop(512),
                NumpyNormalize([0.5], [0.5])
            ])
        }
        return preprocess_image

    def _build_text_transform(self):
        def text_transform(text):
            append_eos = False
            fs_dict = self.dictionary
            if isinstance(self.tokenizer, Encoding):
                words = list(map(str, self.tokenizer.encode(text, allowed_special="all")))
            else:
                words = self.tokenizer.encode(text, out_type=str)
            # ids = [fs_dict.bos_index]
            ids = []
            for i, word in enumerate(words):
                idx = fs_dict.index(word)
                ids.append(idx)
            if append_eos:
                ids.append(fs_dict.eos_index)
            return ids

        return text_transform

    def _batchify(self, lines):
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(
                    lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)

            batches = iterators.BucketedReadaheadBatchIterator(
                lines,
                read_ahead=self.batch_read_ahead,
                key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None,
                batch_size=dynamic_batch_size,
                shuffle=self.shuffle,
                seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)

            gpt_max_length = max([len(x[0]) for x in batch])

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32, fill_value=1)

            chunk_tokens_all = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32, fill_value=0)
            segment_tokens_all = np.full(shape=(batch_size, gpt_max_length - 1), dtype=np.int32, fill_value=0)

            clip_tokens_all = list()
            all_target_image_tokens = []

            for i, (full_tokens, gpt_src_image_tokens, tgt_image_tokens, text_input_mask, text_loss_mask, chunk_tokens,
                    segment_tokens, clip_tokens) in enumerate(batch):
                gpt_source_ids[i, :len(full_tokens) - 1] = full_tokens[:-1]
                gpt_target_ids[i, :len(full_tokens) - 1] = full_tokens[1:]
                gpt_input_mask_all[i, :len(full_tokens) - 1] = text_input_mask[:-1]
                gpt_loss_mask_all[i, :len(full_tokens) - 1] = text_loss_mask[:-1]
                chunk_tokens_all[i, :len(full_tokens) - 1] = chunk_tokens[:-1]
                segment_tokens_all[i, :len(full_tokens) - 1] = segment_tokens[:-1]
                if clip_tokens is not None:
                    clip_tokens_all.append(clip_tokens)
                if tgt_image_tokens is not None:
                    all_target_image_tokens.append(tgt_image_tokens)

            clip_tokens_all = self.clip_tokenizer(
                clip_tokens_all, padding="max_length",
                max_length=self.clip_tokenizer.model_max_length, truncation=True,
                return_tensors="np").input_ids.astype(np.int64) if clip_tokens_all else None
            image_target_ids = np.stack(all_target_image_tokens) if all_target_image_tokens else None

            ret_batch = {
                'vl_laion': {
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'gpt_img_src_tokens': None,
                        'img_tgt_tokens': image_target_ids.astype(np.float32) if image_target_ids is not None else None,
                        'img_gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                        'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                        'chunk_tokens': chunk_tokens_all.astype(np.int64),
                        'segment_tokens': segment_tokens_all.astype(np.int64),
                        'clip_tokens': clip_tokens_all,
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                }
            }

            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches

    def _prepare(self, _random, doc):
        text_tokens = doc[CAPTION_KEY]
        tgt_image_tokens = doc[IMAGE_KEY]
        clip_tokens = doc['clip_tokens']
        text_input_mask = doc['input_mask']
        text_loss_mask = doc['loss_mask']
        chunk_tokens = doc['chunk_tokens']
        segment_tokens = doc['segment_tokens']

        tgt_image_tokens = self.image_transform['diff'](tgt_image_tokens) if tgt_image_tokens is not None else None

        return text_tokens, None, tgt_image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens, clip_tokens

    def _read_from_files(self, source_file):
        file_path = os.path.join(self.data_dir, source_file)
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([])  # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([])  # skip bad file

        bos_id = self.dictionary.bos_index
        eos_id = self.dictionary.eos_index

        for doc_str in lines:
            item = doc_str.strip().split('\t')

            # filter item based self.filter
            # if 'laion2b' in source_file:  # filter out bad image on laion dataset
            is_filter = False
            for filter in self.filters:
                if filter(item):
                    is_filter = True
                    break
            if is_filter:
                continue

            try:
                doc = self.text_transform(item[1])

                if len(doc) > 128:
                    continue

                if self.args.align:
                    clip_tokens = item[1]
                    image_tokens = None
                else:
                    clip_tokens = None
                    image_tokens = Image.open(io.BytesIO(base64.b64decode(item[2]))).convert("RGB")

                text_length = len(doc)

                text_tokens = [bos_id] + doc + [eos_id]
                doc_input_mask = [0] + [0] * text_length + [0]
                doc_loss_mask = [0] + [1] * text_length + [1]
                chunk_tokens = [0] + [0] * text_length + [0]
                segment_tokens = [0] + [0] * text_length + [0]

                yield {
                    CAPTION_KEY: text_tokens,
                    IMAGE_KEY: image_tokens,
                    'input_mask': doc_input_mask,
                    'loss_mask': doc_loss_mask,
                    'chunk_tokens': chunk_tokens,
                    'segment_tokens': segment_tokens,
                    'clip_tokens': clip_tokens,
                }
            except:
                continue
