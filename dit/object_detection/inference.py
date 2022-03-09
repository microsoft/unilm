from argparse import ArgumentParser
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('opts', help='Additional options, such as model weights')

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

