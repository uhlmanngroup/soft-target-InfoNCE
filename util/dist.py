import os
import socket
from pathlib import Path

import torch
from mae.util.misc import setup_for_distributed, save_on_master, is_main_process


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def init_distributed_mode(args):
    args.distributed = True

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    args.dist_url = "tcp://%s:%s" % (master_addr, master_port)

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, top1=None, top5=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        if top1 is not None and top5 is not None:
            # ckpt_name: run_name without timestamp + max_accuracy + 'ckpt' + epoch
            ckpt_name = f'{args.run_name}_top1:{top1:.4f}_top5:{top5:.4f}_epoch:{epoch_name}_ckpt.pth'
        else:
            ckpt_name = f'{args.run_name}_epoch:{epoch_name}_ckpt.pth'

        # checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        checkpoint_paths = [output_dir / ckpt_name]

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if top1 is not None and top5 is not None:
                to_save['top1'] = top1
                to_save['top5'] = top5

            save_on_master(to_save, checkpoint_path)
        assert len(checkpoint_paths) == 1
        return checkpoint_paths[0]
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        ckpt_args = checkpoint.get('args', {})
        assert args.blr == ckpt_args.blr, f"blr {args.blr} != ckpt_args.blr {ckpt_args.blr}"
        assert args.lr == ckpt_args.lr, f"lr {args.lr} != ckpt_args.lr {ckpt_args.lr}"
        assert args.min_lr == ckpt_args.min_lr, f"min_lr {args.min_lr} != ckpt_args.min_lr {ckpt_args.min_lr}"
        assert args.batch_size == ckpt_args.batch_size, f"batch_size {args.batch_size} != ckpt_args.batch_size {ckpt_args.batch_size}"


def remove_on_master(path):
    if path is None or not os.path.exists(path):
        return
    elif is_main_process():
        os.remove(path)

