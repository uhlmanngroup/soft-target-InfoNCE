import random
import torch

from typing import List, Any
from PIL import ImageFilter, ImageOps
from timm.data.mixup import Mixup, mixup_target


def one_hot(x, num_classes, on_value=1., off_value=0.):
    # from timm.data.mixup import one_hot
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def smooth_one_hot(labels, num_classes, smoothing=0.1):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    return one_hot(labels, num_classes=num_classes, on_value=on_value, off_value=off_value)


class NCropsTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, base_transforms: List[Any], num_views: int = 2):
        self.base_transforms = base_transforms
        self.num_views = num_views

    def __call__(self, x):
        """
        Computes self.num_views augmentations randomly sampled
        from self.base_transforms
        """
        return [random.choice(self.base_transforms)(x) for _ in range(self.num_views)]


class OneCropTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transforms: List[Any]):
        assert len(base_transforms) == 2
        self.transform1 = base_transforms[0]
        self.transform2 = base_transforms[1]

    def __call__(self, x):
        transform = random.choice([self.transform1, self.transform2])
        return transform(x)


class TwoCropsTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transforms: List[Any]):
        assert len(base_transforms) == 2
        self.transform1 = base_transforms[0]
        self.transform2 = base_transforms[1]

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class LabelMixup(Mixup):

    def __init__(self, label_smoothing_prob, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing_prob = label_smoothing_prob

    def __call__(self, target):
        assert len(target) % 2 == 0, 'Batch size should be even when using this'
        batch_size = len(target)
        if self.mode == 'elem':
            lam, use_cutmix = self._params_per_elem(batch_size)
        elif self.mode == 'pair':
            lam, use_cutmix = self._params_per_elem(batch_size // 2)
        else:
            lam, use_cutmix = self._params_per_batch()
        
        target = mixup_target(
            target,
            self.num_classes,
            lam,
            self.label_smoothing if random.random() < self.label_smoothing_prob else 0.
        )
        return target


class MixupRandomLabelSmoothing(Mixup):
    def __init__(self, label_smoothing_prob, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing_prob = label_smoothing_prob

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(
            target,
            self.num_classes,
            lam,
            self.label_smoothing if random.random() < self.label_smoothing_prob else 0.
        )
        return x, target