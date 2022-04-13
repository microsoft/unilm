import os
import logging
import functools
import json
import time
from datetime import datetime

# from tensorboardX import SummaryWriter
import yaml
import cv2
import numpy as np

from .config import Configurable, State


class Logger(Configurable):
    SUMMARY_DIR_NAME = 'summaries'
    VISUALIZE_NAME = 'visualize'
    LOG_FILE_NAME = 'output.log'
    ARGS_FILE_NAME = 'args.log'
    METRICS_FILE_NAME = 'metrics.log'

    database_dir = State(default='./outputs/')
    log_dir = State(default='workspace')
    verbose = State(default=False)
    level = State(default='info')
    log_interval = State(default=100)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        self._make_storage()

        cmd = kwargs['cmd']
        self.name = cmd['name']
        self.log_dir = os.path.join(self.log_dir, self.name)
        try:
            self.verbose = cmd['verbose']
        except:
            print('verbose:', self.verbose)
        if self.verbose:
            print('Initializing log dir for', self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.message_logger = self._init_message_logger()

        summary_path = os.path.join(self.log_dir, self.SUMMARY_DIR_NAME)
        self.tf_board_logger = SummaryWriter(summary_path)

        self.metrics_writer = open(os.path.join(
            self.log_dir, self.METRICS_FILE_NAME), 'at')

        self.timestamp = time.time()
        self.logged = -1
        self.speed = None
        self.eta_time = None

    def _make_storage(self):
        application = os.path.basename(os.getcwd())
        storage_dir = os.path.join(
            self.database_dir, self.log_dir, application)
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if not os.path.exists(self.log_dir):
            os.symlink(storage_dir, self.log_dir)

    def save_dir(self, dir_name):
        return os.path.join(self.log_dir, dir_name)

    def _init_message_logger(self):
        message_logger = logging.getLogger('messages')
        message_logger.setLevel(
            logging.DEBUG if self.verbose else logging.INFO)
        formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s] %(message)s')
        std_handler = logging.StreamHandler()
        std_handler.setLevel(message_logger.level)
        std_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, self.LOG_FILE_NAME))
        file_handler.setLevel(message_logger.level)
        file_handler.setFormatter(formatter)

        message_logger.addHandler(std_handler)
        message_logger.addHandler(file_handler)
        return message_logger

    def report_time(self, name: str):
        if self.verbose:
            self.info(name + " time :" + str(time.time() - self.timestamp))
            self.timestamp = time.time()

    def report_eta(self, steps, total, epoch):
        self.logged = self.logged % total + 1
        steps = steps % total
        if self.eta_time is None:
            self.eta_time = time.time()
            speed = -1
        else:
            eta_time = time.time()
            speed = eta_time - self.eta_time
            if self.speed is not None:
                speed = ((self.logged - 1) * self.speed + speed) / self.logged
            self.speed = speed
            self.eta_time = eta_time

        seconds = (total - steps) * speed
        hours = seconds // 3600
        minutes = (seconds - (hours * 3600)) // 60
        seconds = seconds % 60

        print('%d/%d batches processed in epoch %d, ETA: %2d:%2d:%2d' %
              (steps, total, epoch,
               hours, minutes, seconds), end='\r')

    def args(self, parameters=None):
        if parameters is None:
            with open(os.path.join(self.log_dir, self.ARGS_FILE_NAME), 'rt') as reader:
                return yaml.load(reader.read())
        with open(os.path.join(self.log_dir, self.ARGS_FILE_NAME), 'wt') as writer:
            yaml.dump(parameters.dump(), writer)

    def metrics(self, epoch, steps, metrics_dict):
        results = {}
        for name, a in metrics_dict.items():
            results[name] = {'count': a.count, 'value': float(a.avg)}
            self.add_scalar('metrics/' + name, a.avg, steps)
        result_dict = {
            str(datetime.now()): {
                'epoch': epoch,
                'steps': steps,
                **results
            }
        }
        string_result = yaml.dump(result_dict)
        self.info(string_result)
        self.metrics_writer.write(string_result)
        self.metrics_writer.flush()

    def named_number(self, name, num=None, default=0):
        if num is None:
            return int(self.has_signal(name)) or default
        else:
            with open(os.path.join(self.log_dir, name), 'w') as writer:
                writer.write(str(num))
            return num

    epoch = functools.partialmethod(named_number, 'epoch')
    iter = functools.partialmethod(named_number, 'iter')

    def message(self, level, content):
        self.message_logger.__getattribute__(level)(content)

    def images(self, prefix, image_dict, step):
        for name, image in image_dict.items():
            self.add_image(prefix + '/' + name, image, step, dataformats='HWC')

    def merge_save_images(self, name, images):
        for i, image in enumerate(images):
            if i == 0:
                result = image
            else:
                result = np.concatenate([result, image], 0)
        cv2.imwrite(os.path.join(self.vis_dir(), name+'.jpg'), result)

    def vis_dir(self):
        vis_dir = os.path.join(self.log_dir, self.VISUALIZE_NAME)
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        return vis_dir

    def save_image_dict(self, images, max_size=1024):
        for file_name, image in images.items():
            height, width = image.shape[:2]
            if height > width:
                actual_height = min(height, max_size)
                actual_width = int(round(actual_height * width / height))
            else:
                actual_width = min(width, max_size)
                actual_height = int(round(actual_width * height / width))
                image = cv2.resize(image, (actual_width, actual_height))
            cv2.imwrite(os.path.join(self.vis_dir(), file_name+'.jpg'), image)

    def __getattr__(self, name):
        message_levels = set(['debug', 'info', 'warning', 'error', 'critical'])
        if name == '__setstate__':
            raise AttributeError('haha')
        if name in message_levels:
            return functools.partial(self.message, name)
        elif hasattr(self.__dict__.get('tf_board_logger'), name):
            return self.tf_board_logger.__getattribute__(name)
        else:
            super()
