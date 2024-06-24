import os
import glob
import logging
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mae.util import misc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.targets = self.get_labels()

    def get_labels(self):
        labels = []
        for f in self.filenames:
            labels.append(self.id_dict[f.split('/')[4]])
        return labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        with open(img_path, "rb") as f:
            image = PIL.Image.open(f)
            image = image.convert("RGB")
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image)# .type(torch.FloatTensor))
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id] 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        with open(img_path, "rb") as f:
            image = PIL.Image.open(f)
            image = image.convert("RGB")
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image) #.type(torch.FloatTensor))
        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == 'tiny-imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        id_dict = {}
        for i, line in enumerate(open(os.path.join(args.data_path, 'wnids.txt'), 'r')):
          id_dict[line.replace('\n', '')] = i
        if is_train:
            dataset = TrainTinyImageNetDataset(root, id=id_dict, transform=transform)
        else:
            dataset = TestTinyImageNetDataset(root, id=id_dict, transform=transform)
    elif args.dataset == 'cifar100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.CIFAR100(
            root=root,
            train=is_train,
            transform=transform,
            download=True,
        )
    elif args.dataset == 'cifar10':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.CIFAR10(
            root=root,
            train=is_train,
            transform=transform,
            download=True,
        )

    return dataset


def build_transform(is_train, args):
    if args.dataset == 'imagenet': # or args.dataset == 'tiny-imagenet':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif args.dataset == 'tiny-imagenet':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
 
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)

    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )

    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_data(args):
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    logger.info("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            logger.warning('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # get a copy of the labels for training
    train_labels = torch.tensor(dataset_train.targets).clone()
    train_labels = torch.sort(train_labels).values

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_train, data_loader_val, train_labels