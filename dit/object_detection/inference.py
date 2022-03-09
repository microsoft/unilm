import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg()
    # Set config
    cfg.merge_from_file(args.config_file)
    # Set weights using opts argument
    cfg.merge_from_list(args.opts)
    predictor = DefaultPredictor(cfg)
    
    # outputs = predictor(im)

if __name__ == '__main__':
    main()

