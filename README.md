# Noise contrastive estimation with soft targets for conditional models

This repository contains a reference implementation of the loss function and the classification benchmark in
 <a href="https://arxiv.org/abs/2404.14076">noise contrastive estimation with soft targets</a>.

### Image classification experiments
Download the dataset(s) and place them in a folder (e.g. ```datasets/imagenet```).
Download pre-trained vision transformer: <a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">MAE-ViT-B\16</a>.

Baselines can be reproduced with
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_finetune.py --batch_size 128 --finetune {path_to_ViT_checkpoint} --epochs 100 --blr {base_lr_rate} --mixup 0.8 --cutmix 1.0 --mixup_prob 0.8 --smoothing 0.1 --dist_eval --data_path {path_to_data} --output_dir {exp_log_dir} --dataset {dataset_name} --nb_classes {number_of_classes} --world_size {#GPUs} --num_workers {#workers} --loss {loss_name} --mlp_head
```
- ```dataset_name``` can take the values ```{tiny-imagenet, imagenet, cifar100}```
- for running InfoNCE/soft distribution InfoNCE baselines specify ```InfoNCE``` as ```loss_name``` for the targets to be one-hot encoded.
- for running hard target baselines (InfoNCE, NLL) set ```mixup, cutmix, smoothing``` to ```0```

In order to reproduce soft target InfoNCE baselines, set:
- ```blr``` to ```2.5e-4``` for Tiny ImageNet
- ```blr``` to ```5.5e-4``` for ImageNet
- ```blr``` to ```5.5e-4``` for CIFAR-100
