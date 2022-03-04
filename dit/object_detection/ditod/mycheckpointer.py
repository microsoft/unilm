from detectron2.checkpoint import DetectionCheckpointer

from typing import Any
import torch
import torch.nn as nn
from fvcore.common.checkpoint import _IncompatibleKeys, _strip_prefix_if_present, TORCH_VERSION, quantization, \
    ObserverBase, FakeQuantizeBase
from torch import distributed as dist
from scipy import interpolate
import numpy as np
import torch.nn.functional as F


def append_prefix(k):
    prefix = 'backbone.bottom_up.backbone.'
    return prefix + k if not k.startswith(prefix) else k


def modify_ckpt_state(model, state_dict, logger=None):
    # reshape absolute position embedding for Swin
    if state_dict.get(append_prefix('absolute_pos_embed')) is not None:
        absolute_pos_embed = state_dict[append_prefix('absolute_pos_embed')]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.backbone.bottom_up.backbone.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict[append_prefix('absolute_pos_embed')] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    def get_dist_info():
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    rank, _ = get_dist_info()
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            if key not in model.state_dict():
                continue
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.backbone.bottom_up.backbone.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if rank == 0:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.13492:
                #     q = 1.13492

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                if rank == 0:
                    print("x = {}".format(x))
                    print("dx = {}".format(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias

    if append_prefix('pos_embed') in state_dict:
        pos_embed_checkpoint = state_dict[append_prefix('pos_embed')]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.backbone.bottom_up.backbone.patch_embed.num_patches
        num_extra_tokens = model.backbone.bottom_up.backbone.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        new_size_w = model.backbone.bottom_up.backbone.patch_embed.num_patches_w
        new_size_h = model.backbone.bottom_up.backbone.patch_embed.num_patches_h
        # class_token and dist_token are kept unchanged
        if orig_size != new_size_h or orig_size != new_size_w:
            if rank == 0:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size_w, new_size_h))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size_w, new_size_h), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict[append_prefix('pos_embed')] = new_pos_embed

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        if table_key not in model.state_dict():
            continue
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)

    if append_prefix('rel_pos_bias.relative_position_bias_table') in state_dict and \
            model.backbone.bottom_up.backbone.use_rel_pos_bias and \
            not model.backbone.bottom_up.backbone.use_shared_rel_pos_bias and \
            append_prefix('blocks.0.attn.relative_position_bias_table') not in state_dict:
        logger.info("[BEIT] Expand the shared relative position embedding to each transformer block. ")
        num_layers = model.backbone.bottom_up.backbone.get_num_layers()
        rel_pos_bias = state_dict[append_prefix("rel_pos_bias.relative_position_bias_table")]
        for i in range(num_layers):
            state_dict["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        state_dict.pop(append_prefix("rel_pos_bias.relative_position_bias_table"))

    return state_dict


class MyDetectionCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []

        # rename the para in checkpoint_state_dict
        # some bug here, do not support re load

        checkpoint_state_dict = {
            append_prefix(k): checkpoint_state_dict[k]
            for k in checkpoint_state_dict.keys()
        }

        checkpoint_state_dict = modify_ckpt_state(self.model, checkpoint_state_dict, logger=self.logger)

        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                        model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:

                    has_observer_base_classes = (
                            TORCH_VERSION >= (1, 8)
                            and hasattr(quantization, "ObserverBase")
                            and hasattr(quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                                model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            ObserverBase,
                            FakeQuantizeBase,
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )
