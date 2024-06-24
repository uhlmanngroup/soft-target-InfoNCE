import sys
import math
import torch
import wandb
from typing import Iterable, Optional

from torch.nn import CrossEntropyLoss
from timm.data import Mixup
from timm.utils import accuracy
import mae.util.misc as misc
import mae.util.lr_sched as lr_sched
from loss import InfoNCE, SoftTargetInfoNCE, SoftTargetCrossEntropy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, smoothing_fn=None,
                    args=None, **kwargs):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    train_labels = kwargs['train_labels']
    noise_probs = torch.bincount(train_labels) / len(train_labels)
    noise_probs = noise_probs.to(device)

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        elif smoothing_fn is not None:
            assert mixup_fn is None
            targets = smoothing_fn(targets)
        else:
            targets = torch.nn.functional.one_hot(targets, args.nb_classes).float()

        with torch.cuda.amp.autocast():
            logits = model(samples)

            if isinstance(criterion, CrossEntropyLoss) or isinstance(criterion, InfoNCE):
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (data_iter_step + 1) % accum_iter == 0 and misc.is_main_process():
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            wandb.log({'loss': loss_value_reduce, 'lr': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, criterion, args, **kwargs):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        labels = batch[-1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = torch.nn.functional.one_hot(labels, args.nb_classes).float()

        with torch.cuda.amp.autocast():
            logits = model(images)

            if isinstance(criterion, CrossEntropyLoss) or isinstance(criterion, InfoNCE):
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits, targets)

            # class_scores = torch.exp(logits / args.t) if 'NCE' in args.loss else logits / args.t
            class_scores = logits / args.t
            acc1, acc5 = accuracy(class_scores, labels, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        loss_value_reduce = misc.all_reduce_mean(loss.item())
        if misc.is_main_process():
            epoch_1000x = int(epoch) * 1000
            wandb.log({'test/loss': loss_value_reduce}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}