import os
import json
import logging
import datetime
import random
import time
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss
from functools import partial
from pathlib import Path
from timm.data.mixup import Mixup

import mae.util.lr_decay as lrd
from mae.util import misc
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler

from util.data import get_data
from util.default_args import get_default_parser
from util.transforms import smooth_one_hot
from util.vit import get_vit
from util.dist import get_ip_address, get_open_port, init_distributed_mode, remove_on_master, save_model
from train_engine import train_one_epoch, evaluate
from loss import SoftTargetInfoNCE, InfoNCE, SoftTargetCrossEntropy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(rank, world_size, args):
    args.rank = rank
    args.gpu = "cuda:{}".format(rank)
    init_distributed_mode(args)

    logger.info("Rank %d, world_size %d" % (rank, world_size))
    logger.info("Use GPU: {} for training".format(args.gpu))
    logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))

    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_val, train_labels = get_data(args)

    if misc.get_rank() == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)

    # Set up mixup
    mixup_fn = None
    label_mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Set up label smoothing
    smoothing_fn = None
    if not mixup_active and args.smoothing > 0:
        logger.info("Label smoothing is activated!")
        smoothing_fn = partial(smooth_one_hot, num_classes=args.nb_classes, smoothing=args.smoothing)

    if mixup_fn is None and smoothing_fn is None:
        logger.info("Training with Xent! (no mixup and no label smoothing)")
    
    device = torch.device(f"cuda:{rank}")
    model = get_vit(args).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    args.loss_kwargs = {}
    if 'NCE' in args.loss:
        noise_probs = (torch.bincount(train_labels) / len(train_labels)).to(device)
        args.loss_kwargs.update({'t': args.t, 'noise_probs': noise_probs})
    elif args.loss == 'SoftTargetCrossEntropy':
        args.loss_kwargs.update({'t': args.t})

    criterion = eval(args.loss)(**args.loss_kwargs)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if misc.is_main_process():
        logger.info("Model = %s" % str(model_without_ddp))
        logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))
        logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        logger.info("actual lr: %.2e" % args.lr)
        logger.info("accumulate grad iterations: %d" % args.accum_iter)
        logger.info("effective batch size: %d" % eff_batch_size)
        logger.info("criterion: %s" % str(criterion))
        logger.info(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    max_accuracy = 0.0
    val_ckpt = None

    if misc.is_main_process():
        wandb.init(project=args.wandb_project, dir=args.log_dir, config=args, name=args.run_name + '_' + args.timestamp)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            mixup_fn=mixup_fn,
            smoothing_fn=smoothing_fn,
            args=args,
            train_labels=train_labels,
            label_mixup_fn=label_mixup_fn
        )

        if args.output_dir and epoch % 25 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        if epoch % args.eval_freq != 0:
            continue

        test_stats = evaluate(data_loader_val, model, device, epoch, criterion, args)
        if misc.is_main_process():
            logger.info(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            # Remove previous checkpoint and save model with best validation accuracy
            remove_on_master(val_ckpt)
            val_ckpt = save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, top1=test_stats["acc1"], top5=test_stats["acc5"])

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        if misc.is_main_process():
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        if misc.is_main_process():
            wandb.log({'perf/test_acc1': test_stats['acc1'],
                       'perf/test_acc5': test_stats['acc5'],
                       'perf/test_loss': test_stats['loss']},
            )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if misc.is_main_process():
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()

    # set run_name: loss_blr_dataset_seed_datestamp
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.run_name = f'{args.loss}_blr:{args.blr}_{args.dataset}_seed:{args.seed}'
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.run_name + '_' + args.timestamp)
    args.wandb_project = f'ViT-B16_{args.dataset}'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ["MASTER_ADDR"] = get_ip_address()
    os.environ["MASTER_PORT"] = str(get_open_port())

    if args.debug:
        main(0, 1, args)
    else:
        world_size = args.world_size
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
