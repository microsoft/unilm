# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------

from .config import add_vit_config
from .backbone import build_vit_fpn_backbone
from .dataset_mapper import DetrDatasetMapper
from .mycheckpointer import MyDetectionCheckpointer
from .icdar_evaluation import ICDAREvaluator
from .mytrainer import MyTrainer
from .table_evaluation import calc_table_score
from .rcnn_vl import VLGeneralizedRCNN