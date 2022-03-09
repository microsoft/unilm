import argparse
import torch
from ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

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

    # Step 1: set config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    # Step 3: set device
    cfg.MODEL.DEVICE='cpu'
    # Step 4: define model
    model = build_model(cfg)
    model.eval()
    # Step 5: load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    print("Weights loaded!")
    
    # Step 6: run inference
    image = torch.rand(3, 512, 512)
    height, width = image.shape[-2:]
    inputs = {"image": image, "height": height, "width": width}
    
    with torch.no_grad():
        outputs = model([inputs])
        print("Outputs:", outputs)

if __name__ == '__main__':
    main()

