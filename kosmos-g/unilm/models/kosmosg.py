import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import (
    roberta_large_architecture,
    RobertaModel,
)
from fairseq.models.transformer_lm import (
    base_gpt3_architecture,
)
from fairseq.utils import safe_getattr
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from unilm.models.aligner import Aligner, Aligner_encoder
from unilm.models.connector import build_connector
from unilm.models.diffusion import Diffusionmodel, VAE
from unilm.models.gpt import GPTmodel, GPTModelConfig

logger = logging.getLogger(__name__)


def slice_tokens_for_mlm(A, indx, num_elem=2):
    all_indx = indx[:, None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


@dataclass
class KosmosGModelConfig(GPTModelConfig):
    text_encoder: str = field(
        default="none",
        metadata={
            "help": "enable text encoder, options: none, roberta, electra"
        },
    )
    image_encoder: str = field(
        default="clip",
        metadata={
            "help": "enable image encoder, options: none, clip, beit"
        },
    )
    audio_encoder: str = field(
        default="none",
        metadata={
            "help": "enable audio encoder, options: none, "
        },
    )

    # parameters for MLM
    connector: str = field(
        default='xconnector',
        metadata={
            "help": "connector: none, complex, simple, xconnector"
        },
    )
    latent_query_num: int = field(
        default=64,
        metadata={
            "help": "number of latent query tokens"
        },
    )
    remain_tokens: int = field(
        default=300,
        metadata={
            "help": "at least k tokens to produce gpt loss"
        },
    )
    mlm_model_path: str = field(
        default="",
        metadata={"help": "mlm checkpoint path"},
    )
    mlm_dict: str = field(
        default="",
        metadata={"help": "mlm dict path"},
    )
    mlm_tokens_per_sample: int = field(
        default=512,
        metadata={"help": "mlm max length"},
    )

    freeze_gpt: bool = field(
        default=False,
        metadata={
            "help": "freeze gpt parameters"
        },
    )

    # parameters for visual
    visual_model_name: str = field(
        default="ViT-L-14",
        metadata={"help": "model_name for open_clip"}, )
    visual_pretrained: str = field(
        default="",
        metadata={"help": "model_name for visual_pretrained"}, )
    visual_output_dim: int = field(
        default=1024,
        metadata={"help": "output dimension for visual_pretrained"}, )
    no_freeze_layer: str = field(
        default='resblocks.23,ln_post',
        metadata={
            "help": "freeze last layer of visual_pretrained"
        }, )

    # parameters for speech
    speech_model_path: str = field(
        default="",
        metadata={"help": "speech checkpoint path"},
    )
    audio_output_dim: int = field(
        default=768,
        metadata={"help": "output dimension for audio_pretrained"}, )

    # parameters for fine-tuning
    ft_type: int = field(
        default=3,
        metadata={
            "help": "fine-tuning type: \
            1: gpt only \
            2: roberta only \
            3: roberta + gpt \
            4: roberta + gpt(freeze) \
            5: roberta(freeze) + gpt "
        },
    )
    pooler_dropout: float = field(
        default=0.1,
        metadata={"help": "mlm max length"},
    )

    pretrained_ckpt_path: str = field(
        default="",
        metadata={"help": "model checkpoint path"},
    )

    pretrained_model_name_or_path: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "model name or path"},
    )

    align: bool = field(
        default=False,
        metadata={"help": "use clip supervision"},
    )

    lora_dir: str = field(
        default="",
        metadata={"help": "lora dir"},
    )

    lora_name: str = field(
        default="None",
        metadata={"help": "lora name"},
    )


@register_model("kosmosg", dataclass=KosmosGModelConfig)
class KosmosGmodel(BaseFairseqModel):
    def __init__(self, args, gpt_model, diffusion_unet, vae, aligner=None,
                 text_model=None, img_model=None, aud_model=None,
                 text_connector=None, img_connector=None, aud_connector=None,
                 bos=0, eos=2):
        """
        text_model: bidirectional text model, such as roberta, bert, electra
        img_model: image model, such as ViT, CLIP, BEIT
        aud_model: audio model, such as HuBERT, wavLM
        """
        super().__init__()
        self.args = args
        self.gpt_model = gpt_model
        self.diffusion_unet = diffusion_unet
        self.vae = vae
        self.aligner = aligner

        self.text_model = text_model
        self.text_connector = text_connector
        self.img_model = img_model
        self.img_connector = img_connector
        self.aud_model = aud_model
        self.aud_connector = aud_connector

        self.bos = bos
        self.eos = eos
        self.classification_heads = nn.ModuleDict()
        self.ft_type = args.ft_type

    @classmethod
    def build_model(cls, args, task):
        if hasattr(task, "all_dict"):
            task.dictionary = task.all_dict
        if args.align:
            original_checkpoint_activations = args.checkpoint_activations
            args.checkpoint_activations = False
        gpt_model = GPTmodel.build_model(args, task)
        if args.align:
            args.checkpoint_activations = original_checkpoint_activations
        logger.info("gpt args is {}".format(args))

        if args.align:
            aligner = Aligner(args)
            vae = None
            diffusion_unet = None

        else:
            aligner = Aligner_encoder(args)
            vae = VAE.build_model(args, task)
            diffusion_unet = Diffusionmodel.build_model(args, task, 2 ** (len(vae.vae.config.block_out_channels) - 1))

        text_model, text_connector = cls.load_text_model(args, task)
        img_model, img_connector = cls.load_image_model(args, task)
        aud_model, aud_connector = cls.load_audio_model(args, task)

        if args.checkpoint_activations:
            img_model.set_grad_checkpointing(True)

        model = cls(args, gpt_model, diffusion_unet, vae, aligner=aligner,
                    text_model=text_model, text_connector=text_connector,
                    img_model=img_model, img_connector=img_connector,
                    aud_model=aud_model, aud_connector=aud_connector,
                    bos=task.dictionary.bos_index, eos=task.dictionary.eos_index)

        if args.pretrained_ckpt_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_ckpt_path)
            msg = model.load_state_dict(state["model"], strict=False, args=args)
            logger.info(msg)

        cls.freeze_params(model.parameters())
        cls.unfreeze_params(model.aligner.parameters())

        if hasattr(model.aligner, "clip_encoder"):
            cls.freeze_params(model.aligner.clip_encoder.parameters())

        if not args.align:
            cls.unfreeze_params(model.gpt_model.parameters())
            cls.unfreeze_params(model.img_connector.parameters())
            if model.img_model is not None:
                for p_name, p in model.img_model.named_parameters():
                    if args.no_freeze_layer:
                        no_freeze_layers = args.no_freeze_layer.split(',')
                        for no_freeze_layer in no_freeze_layers:
                            if no_freeze_layer in p_name:
                                print("no_freeze_layer: {}".format(p_name))
                                p.requires_grad = True

        cls.stat_params(model)
        return model

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    @staticmethod
    def stat_params(model):
        from tabulate import tabulate
        stat = []
        for n, p in model.named_parameters():
            stat.append([n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["name", "shape", "trainable"]))

    def forward(self, src_tokens,
                mlm_src_tokens=None, gpt_input_mask=None,
                gpt_img_src_tokens=None, img_gpt_input_mask=None, img_tgt_tokens=None,
                aud_src_tokens=None, aud_gpt_input_mask=None,
                gpt_loss_mask=None, mlm_mask=None,
                clip_tokens=None, **kwargs):

        if mlm_src_tokens is not None:
            # mlm
            mlm_output, _ = self.text_model(mlm_src_tokens, features_only=True)
            mlm_output = mlm_output[mlm_mask]
            if self.text_connector is not None:
                # linear projection layer
                mlm_output = self.text_connector(mlm_output)
        else:
            mlm_output = None

        if gpt_img_src_tokens is not None:
            img_output = self.get_image_representation(gpt_img_src_tokens)
        else:
            img_output = None

        if aud_src_tokens is not None:
            aud_output = self.get_audio_representation(aud_src_tokens, kwargs['aud_mask'])
        else:
            aud_output = None

        # gpt
        x, condition, extra = self.gpt_model(src_tokens,
                                             mlm_features=mlm_output, gpt_input_mask=gpt_input_mask,
                                             img_features=img_output, img_gpt_input_mask=img_gpt_input_mask,
                                             aud_features=aud_output, aud_gpt_input_mask=aud_gpt_input_mask,
                                             **kwargs)
        condition = condition.transpose(0, 1)

        if self.args.align:
            extra["loss"] = self.aligner(condition, src_tokens.eq(1), clip_tokens)

        else:
            # diffusion
            latents = self.vae.encode(img_tgt_tokens)
            condition = self.aligner(condition, src_tokens.eq(1))
            null_token = torch.LongTensor([0]).unsqueeze(0).to(condition.device)
            null_token_embeds = self.gpt_model(null_token)[1].transpose(0, 1)
            null_token_embeds = self.aligner(null_token_embeds, null_token.eq(1))

            extra["loss"] = self.diffusion_unet(latents, condition, null_token_embeds)

        # loss mask
        extra["loss_mask"] = gpt_loss_mask
        return x, extra

    def encode_condition(self, src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                         num_images_per_prompt):
        model_device = next(self.gpt_model.parameters()).device
        if src_tokens.device != model_device:
            src_tokens = src_tokens.to(model_device) if src_tokens is not None else None
            gpt_img_src_tokens = gpt_img_src_tokens.to(model_device) if gpt_img_src_tokens is not None else None
            img_gpt_input_mask = img_gpt_input_mask.to(model_device) if img_gpt_input_mask is not None else None
            negative_tokens = negative_tokens.to(model_device) if negative_tokens is not None else None

        if gpt_img_src_tokens is not None:
            img_output = self.get_image_representation(gpt_img_src_tokens)
        else:
            img_output = None

        condition = self.gpt_model(src_tokens, img_features=img_output, img_gpt_input_mask=img_gpt_input_mask)[
            1].transpose(0, 1)
        condition = self.aligner(condition, src_tokens.eq(1))

        null_token = torch.LongTensor([0]).unsqueeze(0).to(condition.device)
        null_token = max(null_token, negative_tokens, key=lambda x: x.shape[1])
        null_token_embeds = self.gpt_model(null_token)[1].transpose(0, 1)
        null_token_embeds = self.aligner(null_token_embeds, null_token.eq(1))

        condition = torch.cat([
            null_token_embeds.repeat(len(condition) * num_images_per_prompt, 1, 1),
            condition.repeat(num_images_per_prompt, 1, 1)
        ])
        return condition

    def sample(self, src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, **kwargs):
        condition = self.encode_condition(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                          kwargs['num_images_per_prompt'])

        latents = self.diffusion_unet.sample(condition, **kwargs)
        image = self.vae.decode(latents, output_type=kwargs['output_type'] if 'output_type' in kwargs else 'pil')
        return image

    def sample_controlnet(self, src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens, control_image,
                          controlnet, **kwargs):
        condition = self.encode_condition(src_tokens, gpt_img_src_tokens, img_gpt_input_mask, negative_tokens,
                                          kwargs['num_images_per_prompt'])
        latents = self.diffusion_unet.sample_controlnet(condition, control_image, controlnet, **kwargs)
        image = self.vae.decode(latents, output_type=kwargs['output_type'] if 'output_type' in kwargs else 'pil')
        return image

    def get_image_representation(self, gpt_img_src_tokens):
        # image
        if self.args.image_encoder == "clip":
            img_output = self.img_model(gpt_img_src_tokens)
        elif self.args.image_encoder == "b3-large":
            img_output = \
                self.img_model(textual_tokens=None, visual_tokens=gpt_img_src_tokens, vision_masked_position=None)[
                    "encoder_out"]
            img_output = F.normalize(img_output, dim=-1)
        elif self.args.image_encoder == "b3-base":
            img_output = \
                self.img_model(textual_tokens=None, visual_tokens=gpt_img_src_tokens, vision_masked_position=None)[
                    "encoder_out"]
            img_output = F.normalize(img_output, dim=-1)
        src_len = img_output.size(0)
        img_output = img_output.transpose(0, 1)  # T x B x C -> B x T x C
        img_output = img_output.reshape(-1, img_output.size(-1))

        if self.img_connector is not None:
            # linear projection layer
            img_output = self.img_connector(img_output, src_len=src_len)
        return img_output

    def get_audio_representation(self, aud_src_tokens, aud_mask):
        # audio
        if len(aud_src_tokens.size()) == 3:
            aud_src_tokens = aud_src_tokens.unsqueeze(0)
            aud_mask = aud_mask.unsqueeze(0)
        # aud_src_tokens  B * Seg size * Fbank size * Dim
        fbank = aud_src_tokens.view(-1, aud_src_tokens.size(2), aud_src_tokens.size(3))
        # B * Seg size * Fbank size * Dim -> B * Seg size * Fbank size * Dim
        # aud_mask  B * Seg size * Token size
        padding_mask = aud_mask.view(-1, aud_mask.size(2))  # B * Seg size * Token size -> B * Seg size * Token size
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, 2).view(-1, fbank.size(1))
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, fbank.size(-1))
        aud_output = self.aud_model.extract_features(source=None,
                                                     fbank=fbank,
                                                     padding_mask=padding_mask)[0]
        aud_output = aud_output[~aud_mask.view(-1, aud_mask.size(-1))]
        if self.aud_connector is not None:
            # linear projection layer
            aud_output = self.aud_connector(aud_output, src_len=aud_src_tokens.size(2))
        return aud_output

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args.ft_type
        )

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def supported_targets(self):
        return {"future"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

    @classmethod
    def load_text_model(cls, args, task):
        """Load a roberta model from the fairseq library."""
        if args.text_encoder == "none":
            return None, None
        mlm_args = copy.deepcopy(args)
        mlm_task = task
        logger.info("Roberta dictionary: {} types".format(len(mlm_task.dictionary)))

        mlm_args.layernorm_embedding = True
        mlm_args.no_scale_embedding = True
        mlm_args.dropout = 0.1
        mlm_args.attention_dropout = 0.1
        mlm_args.tokens_per_sample = mlm_args.mlm_tokens_per_sample
        mlm_model = RobertaModel.build_model(mlm_args, mlm_task)
        logger.info("mlm args is {}".format(mlm_args))
        if args.mlm_model_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.mlm_model_path)
            mlm_model.load_state_dict(state["model"], strict=True, args=mlm_args)
        connector = build_connector(args, args.encoder_embed_dim, args.decoder_embed_dim)
        return mlm_model, connector

    @classmethod
    def load_image_model(cls, args, task):
        if args.image_encoder == "none":
            return None, None
        if args.image_encoder == "clip":
            from unilm.models.vl.clip import create_model
            force_quick_gelu = False
            if args.visual_model_name == "ViT-L-14":
                force_quick_gelu = True
            model = create_model(args.visual_model_name, pretrained=args.visual_pretrained,
                                 force_quick_gelu=force_quick_gelu)
            connector = build_connector(args, args.visual_output_dim, args.decoder_embed_dim)
            return model, connector
        if args.image_encoder == "b3-large":
            drop_path_rate = 0.0
            embed_dim = 1024
            depth = 24
            num_heads = 16
            mlp_ratio = 4
            encoder_args = EncoderConfig(vocab_size=64010, multiway=True, layernorm_embedding=False,
                                         no_output_layer=True, drop_path_rate=drop_path_rate,
                                         encoder_embed_dim=embed_dim, encoder_attention_heads=num_heads,
                                         encoder_ffn_embed_dim=int(embed_dim * mlp_ratio), encoder_layers=depth)
            encoder = BEiT3(encoder_args)
            encoder.load_state_dict(torch.load(args.visual_pretrained, map_location='cpu')["module"], strict=True)
            model = encoder
            connector = build_connector(args, args.visual_output_dim, args.decoder_embed_dim)
            return model, connector
        if args.image_encoder == "b3-base":
            drop_path_rate = 0.0
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4
            encoder_args = EncoderConfig(vocab_size=64010, multiway=True, layernorm_embedding=False,
                                         no_output_layer=True, drop_path_rate=drop_path_rate,
                                         encoder_embed_dim=embed_dim, encoder_attention_heads=num_heads,
                                         encoder_ffn_embed_dim=int(embed_dim * mlp_ratio), encoder_layers=depth)
            encoder = BEiT3(encoder_args)
            # trim encoder in model state_dict
            state_dict = {}
            old_state_dict = torch.load(args.visual_pretrained, map_location='cpu')["module"]
            for key in old_state_dict.keys():
                new_key = key
                if 'head' in key:
                    continue
                if 'encoder.encoder' in new_key:
                    new_key = new_key.replace('encoder.encoder', 'encoder')
                elif 'encoder.text_embed' in new_key:
                    new_key = new_key.replace('encoder.text_embed', 'text_embed')
                elif 'encoder.vision_embed' in new_key:
                    new_key = new_key.replace('encoder.vision_embed', 'vision_embed')

                state_dict[new_key] = old_state_dict[key]

            encoder.load_state_dict(state_dict, strict=True)
            model = encoder
            connector = build_connector(args, args.visual_output_dim, args.decoder_embed_dim)
            return model, connector

    @classmethod
    def load_audio_model(cls, args, task):
        if args.audio_encoder == "none":
            return None, None
        if args.audio_encoder == "wavlm":
            from unilm.models.speech.WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(args.speech_model_path)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
            connector = build_connector(args, args.audio_output_dim, args.decoder_embed_dim)
            return model, connector
        return None, None


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
            ft_type
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.ft_type = ft_type

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("kosmosg", "kosmosg_xl")
def kosmosg_xl(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)
