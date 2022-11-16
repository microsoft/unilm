# coding=utf-8
from transformers.utils import logging

from ..layoutlmv2 import LayoutLMv2Config


logger = logging.get_logger(__name__)

LAYOUTXLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutxlm-base": "https://huggingface.co/layoutxlm-base/resolve/main/config.json",
    "layoutxlm-large": "https://huggingface.co/layoutxlm-large/resolve/main/config.json",
}


class LayoutXLMConfig(LayoutLMv2Config):
    model_type = "layoutxlm"
