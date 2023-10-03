import argparse
import gc
import os

from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC

from tiktoken.core import Encoding
from torchvision.transforms import CenterCrop, Compose, Resize
from diffusers import ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler

from app_utils import *
from controlnet.preprocessor import ControlNet_Preprocessor
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from unilm.data.vl.openimage_loader import NumpyNormalize


class AppModel:
    def __init__(self, cfg):
        if isinstance(cfg, argparse.Namespace):
            cfg = convert_namespace_to_omegaconf(cfg)
        cfg.model.align = False
        cfg.model.checkpoint_activations = False
        utils.import_user_module(cfg.common)
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)
        model.freeze_params(model.parameters())
        model.half()
        model.cuda()
        model.eval()

        self.cfg = cfg
        self.model = model
        self.model_cache = {}
        self.bos_id = task.dictionary.bos()
        self.eos_id = task.dictionary.eos()
        self.boi_id = task.dictionary.index(BOI_SYMBOL)
        self.eoi_id = task.dictionary.index(EOI_SYMBOL)
        self.dictionary = task.dictionary
        self.tokenizer = task.tokenizer
        self.text_transform = self._build_text_transform()
        self.image_transform = self._build_image_transform()
        self.task_name = ""
        self.controlnet = None
        self.controlnet_preprocessor = ControlNet_Preprocessor()

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

    def load_controlnet_weight(self, task_name):
        if task_name == self.task_name:
            return

        torch.cuda.empty_cache()
        gc.collect()
        self.controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_IDS[task_name], torch_dtype=torch.float16)
        self.model.freeze_params(self.controlnet.parameters())
        self.controlnet.cuda()
        self.controlnet.eval()
        torch.cuda.empty_cache()
        gc.collect()
        self.task_name = task_name

    def get_available_lora(self):
        # traverse all the available lora in cfg.model.lora_dir in created time order
        if self.cfg.model.lora_dir == '':
            return []
        files = [f for f in os.listdir(self.cfg.model.lora_dir) if
                 os.path.isfile(os.path.join(self.cfg.model.lora_dir, f))]
        files_sorted = sorted(files, key=lambda x: os.path.getctime(os.path.join(self.cfg.model.lora_dir, x)))

        return files_sorted

    def load_lora(self, lora_name):
        if lora_name == 'None':
            if self.cfg.model.lora_name != 'None':
                for _, module in self.model.diffusion_unet.unet.named_modules():
                    if hasattr(module, "set_lora_layer"):
                        module.set_lora_layer(None)
                self.model.diffusion_unet.lora = False
            self.cfg.model.lora_name = 'None'
            return 'None'
        try:
            state_dict, network_alphas = self.model.diffusion_unet.lora_state_dict(
                os.path.join(self.cfg.model.lora_dir, lora_name)
            )

            self.model.diffusion_unet.load_lora_into_unet(
                state_dict, network_alphas=network_alphas, unet=self.model.diffusion_unet.unet, low_cpu_mem_usage=True
            )
            self.model.diffusion_unet.lora = True
            self.cfg.model.lora_name = lora_name
            return lora_name
        except:
            return 'None'

    def set_ckpt_scheduler_fn(self, ckpt, scheduler):
        # reset scheduler if the class is changed
        if scheduler == 'dpms' and not isinstance(self.model.diffusion_unet.scheduler, DPMSolverMultistepScheduler):
            self.model.diffusion_unet.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16,
                revision="fp16"
            )
        elif scheduler == 'pndm' and not isinstance(self.model.diffusion_unet.scheduler, PNDMScheduler):
            self.model.diffusion_unet.scheduler = PNDMScheduler.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16,
                revision="fp16"
            )
        elif scheduler == 'ddim' and not isinstance(self.model.diffusion_unet.scheduler, DDIMScheduler):
            self.model.diffusion_unet.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16,
                revision="fp16"
            )

        if ckpt != self.cfg.model.pretrained_ckpt_path:
            try:
                if ckpt not in self.model_cache:
                    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
                    self.model_cache[ckpt] = state["model"]
                msg = self.model.load_state_dict(self.model_cache[ckpt], strict=False, args=self.cfg.model)
                self.cfg.model.pretrained_ckpt_path = ckpt

            except:
                if ckpt in self.model_cache:
                    del self.model_cache[ckpt]
        exp = self.cfg.model.pretrained_ckpt_path.split('/')[-2]
        pt = self.cfg.model.pretrained_ckpt_path.split('/')[-1].split('.')[0].split('_')[-1]
        return f'exp: {exp} pt: {pt}'

    def kosmosg_preprocess(self, prompt, negative_prompt, *args, single_batch=True):
        img_src_tokens = [im for im in args if im is not None]
        assert len(img_src_tokens) == prompt.count('<i>'), \
            "Number of images in prompt does not match the number of images uploaded"

        gpt_img_src_tokens = [torch.tensor(self.image_transform['gpt'](im)) for im in img_src_tokens]

        src_tokens = [self.bos_id]
        img_gpt_input_mask = [0]

        for i in range(len(img_src_tokens)):
            text_snippet = prompt.split('<i>', 1)[0]
            prompt = prompt.split('<i>', 1)[1]
            text_token = self.text_transform(text_snippet)

            src_tokens.extend(text_token + [self.boi_id] * (self.cfg.task.image_token_length + 1) + [self.eoi_id])
            img_gpt_input_mask.extend([0] * len(text_token) + [0] + [1] * self.cfg.task.image_token_length + [0])

        text_token = self.text_transform(prompt)
        src_tokens.extend(text_token)
        img_gpt_input_mask.extend([0] * len(text_token))

        src_tokens = torch.LongTensor(src_tokens)
        gpt_img_src_tokens = torch.stack(gpt_img_src_tokens).to(torch.float16) \
            if len(gpt_img_src_tokens) > 0 else None
        img_gpt_input_mask = torch.tensor(img_gpt_input_mask, dtype=torch.bool)

        negative_tokens = torch.LongTensor([self.bos_id] + self.text_transform(negative_prompt))

        if single_batch:
            return src_tokens.unsqueeze(0), gpt_img_src_tokens, img_gpt_input_mask.unsqueeze(0), \
                   negative_tokens.unsqueeze(0)
        else:
            return src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens

    @torch.inference_mode()
    def kosmosg_generation(self, prompt, lora_scale, num_inference_steps, text_guidance_scale, negative_prompt,
                           num_images_per_prompt, *args):
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': lora_scale,
        }

        image = self.model.sample(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, **kwargs)
        return image

    @torch.inference_mode()
    def controlnet_generation_canny(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                    num_images_per_prompt, control_image, image_resolution, low_threshold,
                                    high_threshold, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_canny(control_image, image_resolution, low_threshold,
                                                                      high_threshold)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('Canny')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_mlsd(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                   num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                   value_threshold, distance_threshold, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_mlsd(control_image, image_resolution,
                                                                     preprocess_resolution, value_threshold,
                                                                     distance_threshold)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('MLSD')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_scribble(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                       num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                       preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_scribble(control_image, image_resolution,
                                                                         preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('scribble')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_scribble_interactive(self, prompt, num_inference_steps, text_guidance_scale,
                                                   negative_prompt, num_images_per_prompt, control_image_and_mask,
                                                   image_resolution, *args):
        assert control_image_and_mask is not None, 'Image is required'

        control_image = self.controlnet_preprocessor.preprocess_scribble_interactive(control_image_and_mask,
                                                                                     image_resolution)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('scribble')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_softedge(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                       num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                       preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_softedge(control_image, image_resolution,
                                                                         preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('softedge')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_openpose(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                       num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                       preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_openpose(control_image, image_resolution,
                                                                         preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('Openpose')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_segmentation(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                           num_images_per_prompt, control_image, image_resolution,
                                           preprocess_resolution, preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_segmentation(control_image, image_resolution,
                                                                             preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('segmentation')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_depth(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                    num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                    preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_depth(control_image, image_resolution,
                                                                      preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('depth')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_normal(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                     num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                     preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_normal(control_image, image_resolution,
                                                                       preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('NormalBae')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_lineart(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                      num_images_per_prompt, control_image, image_resolution, preprocess_resolution,
                                      preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_lineart(control_image, image_resolution,
                                                                        preprocess_resolution, preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        if 'anime' in preprocessor_name:
            self.load_controlnet_weight('lineart_anime')
        else:
            self.load_controlnet_weight('lineart')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, **kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_shuffle(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                      num_images_per_prompt, control_image, image_resolution, preprocessor_name, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_shuffle(control_image, image_resolution,
                                                                        preprocessor_name)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('shuffle')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, kwargs)
        return [control_image] + image

    @torch.inference_mode()
    def controlnet_generation_ip2p(self, prompt, num_inference_steps, text_guidance_scale, negative_prompt,
                                   num_images_per_prompt, control_image, image_resolution, *args):
        assert control_image is not None, 'Image is required'
        assert image_resolution <= MAX_IMAGE_RESOLUTION, 'Image resolution is too high'

        control_image = self.controlnet_preprocessor.preprocess_ip2p(control_image, image_resolution)
        src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens = \
            self.kosmosg_preprocess(prompt, negative_prompt, *args)

        self.load_controlnet_weight('ip2p')

        kwargs = {
            'num_inference_steps': num_inference_steps,
            'text_guidance_scale': text_guidance_scale,
            'num_images_per_prompt': num_images_per_prompt,
            'lora_scale': 0.0,
        }

        image = self.model.sample_controlnet(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                             control_image, self.controlnet, kwargs)
        return [control_image] + image
