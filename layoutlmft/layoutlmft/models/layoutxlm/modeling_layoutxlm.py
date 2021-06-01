# coding=utf-8
from transformers.utils import logging

from ..layoutlmv2 import LayoutLMv2ForRelationExtraction, LayoutLMv2ForTokenClassification, LayoutLMv2Model
from .configuration_layoutxlm import LayoutXLMConfig


logger = logging.get_logger(__name__)

LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutxlm-base",
    "layoutxlm-large",
]


class LayoutXLMModel(LayoutLMv2Model):
    config_class = LayoutXLMConfig


class LayoutXLMForTokenClassification(LayoutLMv2ForTokenClassification):
    config_class = LayoutXLMConfig


class LayoutXLMForRelationExtraction(LayoutLMv2ForRelationExtraction):
    config_class = LayoutXLMConfig
