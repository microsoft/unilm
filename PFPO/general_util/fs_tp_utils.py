# Copied from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/core/tensor_parallel/data.py#L75

from fairscale.nn.model_parallel import initialize as mpu



def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    if get_sequence_parallel_world_size() > 1:
        rank = get_sequence_parallel_rank()
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
    else:
        rank = get_tensor_model_parallel_rank()
        src_rank = get_tensor_model_parallel_src_rank()
        group = get_tensor_model_parallel_group()

    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
        keys, data, group=group, rank=rank, src_rank=src_rank)

    # Pack on rank zero.
    if rank == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0).to(get_accelerator().device_name())
    else:
        flatten_data = torch.empty(total_numel,
                                   device=get_accelerator().current_device_name(),
                                   dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(flatten_data, src_rank, group=group)

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
