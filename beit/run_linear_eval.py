# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on DINO code bases
# https://github.com/facebookresearch/dino/blob/main/eval_linear.py
# --------------------------------------------------------'
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import modeling_finetune
from timm.models import create_model


def load_model(model, checkpoint_file, model_key, model_prefix):
    if checkpoint_file.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_file, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    checkpoint_model = None
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint

    utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)


def eval_linear(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    mean = (0.485, 0.456, 0.406) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    print("train_transform = %s" % str(train_transform))
    print("val_transform = %s" % str(val_transform))
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    global_rank = utils.get_rank()
    world_size = utils.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=world_size, rank=global_rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = create_model(
        args.model, pretrained=False, num_classes=0, drop_rate=0, drop_path_rate=0,
        attn_drop_rate=0, drop_block_rate=None, use_mean_pooling=False,
        use_shared_rel_pos_bias=args.rel_pos_bias, use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    model.cuda()
    model.eval()
    print(f"Model {args.model} built.")
    # load weights to evaluate
    load_model(model=model, checkpoint_file=args.pretrained_weights, model_key=args.checkpoint_key, model_prefix="")

    linear_classifier = LinearClassifier(
        dim=model.embed_dim * (1 + int(args.avgpool_patchtokens)), 
        num_labels=args.num_labels, num_layers=model.get_num_layers())
    linear_classifier = linear_classifier.cuda()
    if world_size > 1:
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    print("Model = %s" % str(linear_classifier))

    # set optimizer
    learning_rate = args.lr or args.base_lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256
    # use absolute or linear scaled learning rate
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(), learning_rate, momentum=0.9, 
            weight_decay=0, # we do not apply weight decay
        )
    else:
        optimizer = torch.optim.AdamW(
            linear_classifier.parameters(), learning_rate, weight_decay=1e-4, 
        )
    print(f"Optimizer = %s" % str(optimizer))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(
            model, linear_classifier, optimizer, train_loader, epoch, args.avgpool_patchtokens, args.amp_forward)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(
                val_loader, model, linear_classifier, args.avgpool_patchtokens, args.amp_forward)
            for classifier_key in test_stats:
                classifier = test_stats[classifier_key]
                print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {classifier['acc1']:.1f}%")
                best_acc = max(best_acc, classifier["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, avgpool, amp_forward):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    assert avgpool
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if amp_forward:
                with torch.cuda.amp.autocast():
                    intermediate_output = model.get_intermediate_layers(inp)
            else:
                intermediate_output = model.get_intermediate_layers(inp)

            output = []
            for each_layer in intermediate_output:
                cls_rep = each_layer[:, 0]
                mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                output.append(torch.cat((cls_rep, mean_rep), dim=-1).float())

        output = linear_classifier(output)

        # compute cross entropy loss
        loss = 0
        for each_output in output:
            loss += nn.CrossEntropyLoss()(each_output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, avgpool, amp_forward):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    assert avgpool
    module = linear_classifier.module if hasattr(linear_classifier, 'module') else linear_classifier
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if amp_forward:
                with torch.cuda.amp.autocast():
                    intermediate_output = model.get_intermediate_layers(inp)
            else:
                intermediate_output = model.get_intermediate_layers(inp)

            output = []
            for each_layer in intermediate_output:
                cls_rep = each_layer[:, 0]
                mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                output.append(torch.cat((cls_rep, mean_rep), dim=-1).float())
            all_output = linear_classifier(output)

        for i, output in enumerate(all_output):
            loss = nn.CrossEntropyLoss()(output, target)

            if module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            post_str = '_layer%d' % i
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1' + post_str].update(acc1.item(), n=batch_size)
            if module.num_labels >= 5:
                metric_logger.meters['acc5' + post_str].update(acc5.item(), n=batch_size)

    eval_results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    updated_results = {}
    for key in eval_results:
        if '_' in key:
            this_key, classifier_idx = key.split('_')

            if classifier_idx not in updated_results:
                updated_results[classifier_idx] = {}

            updated_results[classifier_idx][this_key] = eval_results[key]

    print("Eval result = %s" % json.dumps(updated_results, indent=2))
    return updated_results


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_layers, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.ModuleList()
        self.num_classifier = num_layers
        for i in range(self.num_classifier):
            linear = nn.Linear(dim, num_labels)
            linear.weight.data.normal_(mean=0.0, std=0.01)
            linear.bias.data.zero_()
            self.linear.append(linear)

    def forward(self, x_list):
        results = []
        for i, linear in enumerate(self.linear):
            results.append(linear(x_list[i]))
        return results


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--avgpool_patchtokens', default=True, type=bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to True for BEiT pretrained models. """)
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--optimizer', default="adamw", type=str, help='optimizer type')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="model|module|teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument("--base_lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, type=bool_flag,
        help="""Set True to use the imagenet default mean and std, Set False will use the mean and std in Inception. 
        We recommand keep it same to the pre-training stage. """)
    parser.add_argument('--amp_forward', default=True, type=bool_flag, help='Use amp to inference the pre-trained model, which can speed up the evaluation. ')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
