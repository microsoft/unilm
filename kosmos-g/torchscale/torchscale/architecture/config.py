# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


class EncoderConfig(object):
    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_encoder_input_output_embed = kwargs.pop(
            "share_encoder_input_output_embed", False
        )
        self.max_source_positions = kwargs.pop("max_source_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Vision
        self.img_size = kwargs.pop("img_size", 224)
        self.patch_size = kwargs.pop("patch_size", 16)
        self.in_chans = kwargs.pop("in_chans", 3)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.scale_length = kwargs.pop("scale_length", 2048)
        # Lora
        self.lora = kwargs.pop("lora", False)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


class DecoderConfig(object):
    def __init__(self, **kwargs):
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 12)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 3072)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", False
        )
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.sope_rel_pos = kwargs.pop("sope_rel_pos", False)
        self.scale_length = kwargs.pop("scale_length", 2048)
        # Lora
        self.lora = kwargs.pop("lora", False)

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


class EncoderDecoderConfig(object):
    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 12)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 3072)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_all_embeddings = kwargs.pop("share_all_embeddings", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", False
        )
        self.max_source_positions = kwargs.pop("max_source_positions", 1024)
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.sope_rel_pos = kwargs.pop("sope_rel_pos", False)
        self.scale_length = kwargs.pop("scale_length", 2048)
        # Lora
        self.lora = kwargs.pop("lora", False)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
