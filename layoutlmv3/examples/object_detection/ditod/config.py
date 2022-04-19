from detectron2.config import CfgNode as CN


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    _C = cfg

    _C.MODEL.VIT = CN()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"

    _C.MODEL.VIT.DROP_PATH = 0.

    _C.MODEL.VIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _C.AUG = CN()

    _C.AUG.DETR = False

    _C.MODEL.IMAGE_ONLY = True
    _C.PUBLAYNET_DATA_DIR_TRAIN = ""
    _C.PUBLAYNET_DATA_DIR_TEST = ""
    _C.ICDAR_DATA_DIR_TRAIN = ""
    _C.ICDAR_DATA_DIR_TEST = ""
    _C.CACHE_DIR = ""
    _C.MODEL.CONFIG_PATH = ""

    # effective update steps would be MAX_ITER/GRADIENT_ACCUMULATION_STEPS
    # maybe need to set MAX_ITER *= GRADIENT_ACCUMULATION_STEPS
    _C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
