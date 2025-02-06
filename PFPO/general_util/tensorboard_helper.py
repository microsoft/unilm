from typing import Union, Dict, Tuple, Any

from torch import Tensor
# from torch.utils.tensorboard import SummaryWriter

from general_util.logger import get_child_logger

logger = get_child_logger("TensorboardHelper")


class SummaryWriterHelper:
    def __init__(self,
                 # writer: SummaryWriter,
                 writer,
                 batch_index_or_keys: Dict[str, Union[int, str]] = None,
                 outputs_index_or_keys: Dict[str, Union[int, str]] = None):
        """
        :param writer:
        :param batch_index_or_keys: use key to support dict and index (int) to support tuple.
        :param outputs_index_or_keys: use key to support dict and index (int) to support tuple.
        """
        self.writer = writer
        self.batch_index_or_keys = batch_index_or_keys
        self.outputs_index_or_keys = outputs_index_or_keys
        logger.info("Tensorboard details:")
        logger.info(self.batch_index_or_keys)
        logger.info(self.outputs_index_or_keys)

    def __call__(self, step: int, last_batch: Union[Dict, Tuple] = None, last_outputs: Union[Dict, Tuple] = None):
        if last_batch is not None and self.batch_index_or_keys is not None:
            for name, k in self.batch_index_or_keys.items():
                if last_batch[k] is not None:
                    if isinstance(last_batch[k], Tensor):
                        scalar = last_batch[k].item()
                    else:
                        scalar = last_batch[k]
                    self.writer.add_scalar(name, scalar, global_step=step)
        if last_outputs is not None and self.outputs_index_or_keys is not None:
            for name, k in self.outputs_index_or_keys.items():
                if last_outputs[k] is not None:
                    if isinstance(last_outputs[k], Tensor):
                        scalar = last_outputs[k].item()
                    else:
                        scalar = last_outputs[k]
                    self.writer.add_scalar(name, scalar, global_step=step)


class WandbWriter:
    def __init__(self,
                 batch_index_or_keys: Dict[str, Union[int, str]] = None,
                 outputs_index_or_keys: Dict[str, Union[int, str]] = None):
        """
        :param batch_index_or_keys: use key to support dict and index (int) to support tuple.
        :param outputs_index_or_keys: use key to support dict and index (int) to support tuple.
        """
        self.batch_index_or_keys = batch_index_or_keys
        self.outputs_index_or_keys = outputs_index_or_keys
        logger.info("Logs details:")
        logger.info(self.batch_index_or_keys)
        logger.info(self.outputs_index_or_keys)

        self.logs = {}
        self.logs_accumulation = {}

    def update(self, last_batch: Union[Dict, Tuple] = None, last_outputs: Union[Dict, Tuple] = None):
        if last_batch is not None and self.batch_index_or_keys is not None:
            for name, k in self.batch_index_or_keys.items():
                if last_batch[k] is not None:
                    if isinstance(last_batch[k], Tensor):
                        scalar = last_batch[k].item()
                    else:
                        scalar = last_batch[k]
                    if name not in self.logs:
                        self.logs[name] = scalar
                        self.logs_accumulation[name] = 1
                    else:
                        self.logs[name] += scalar
                        self.logs_accumulation[name] += 1
        if last_outputs is not None and self.outputs_index_or_keys is not None:
            for name, k in self.outputs_index_or_keys.items():
                if last_outputs[k] is not None:
                    if isinstance(last_outputs[k], Tensor):
                        scalar = last_outputs[k].item()
                    else:
                        scalar = last_outputs[k]
                    if name not in self.logs:
                        self.logs[name] = scalar
                        self.logs_accumulation[name] = 1
                    else:
                        self.logs[name] += scalar
                        self.logs_accumulation[name] += 1

    def __call__(self, clear: bool = True) -> Dict[str, Any]:
        logs = {k: v / self.logs_accumulation[k] for k, v in self.logs.items()}
        if clear:
            self.logs = {}
            self.logs_accumulation = {}
        return logs
