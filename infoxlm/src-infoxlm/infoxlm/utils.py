import torch

from fairseq import utils

@torch.no_grad()
def concat_all_gather(tensor):
  """
  Performs all_gather operation on the provided tensors.
  *** Warning ***: torch.distributed.all_gather has no gradient.
  """
  if torch.cuda.device_count() > 1:
    return varsize_tensor_all_gather(tensor)
  else:
    output = tensor
  return output

@torch.no_grad()
def tensor_all_gather(tensor):
  tensors_gather = [torch.ones_like(tensor)
    for _ in range(torch.distributed.get_world_size())]
  torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

  output = torch.cat(tensors_gather, dim=0)
  return output

@torch.no_grad()
def varsize_tensor_all_gather(tensor):

  # cuda_device = f'cuda:{torch.distributed.get_rank()}
  cuda_device = 'cuda'
  if tensor is None:
    size_tens = torch.tensor([0], dtype=torch.int64, device=cuda_device)
  else:
    size_tens = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=cuda_device)

  # print("size_tens", flush=True)
  # print(size_tens, flush=True)
  size_tens = tensor_all_gather(size_tens).cpu()

  max_size = size_tens.max()

  padded = torch.empty(max_size, *tensor.shape[1:],
                        dtype=tensor.dtype,
                        device=cuda_device)
  if tensor is not None:
    padded[:tensor.shape[0]] = tensor
  # print("padded:", flush=True)
  # print(padded, flush=True)

  ag = tensor_all_gather(padded)
  # print("ag:", flush=True)
  # print(ag, flush=True)

  slices = []
  for i, sz in enumerate(size_tens):
      start_idx = i * max_size
      end_idx = start_idx + sz.item()

      if end_idx > start_idx:
          slices.append(ag[start_idx:end_idx])

  ret = torch.cat(slices, dim=0)

  return ret.to(tensor)


def _get_logging_loss(loss, reduce=True):
  if loss is None: return 0
  return utils.item(loss.data) if reduce else loss.data


def construct_idx_tensor_from_list(idx_list2d, lens, pad_idx, device=None):
  max_len = max(lens)
  padded_list = [list_i + [pad_idx] * (max_len - lens[i]) for i, list_i in enumerate(idx_list2d)]
  tensor = torch.LongTensor(padded_list)
  if device is not None:
    tensor = tensor.to(device=device)
  return tensor


def move_to_device(sample, device):

  def _move_to_device(tensor):
    return tensor.to(device=device)
  
  return utils.apply_to_sample(_move_to_device, sample)
