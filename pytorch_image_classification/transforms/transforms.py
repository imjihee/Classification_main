from typing import Tuple, Union

import numpy as np
import PIL.Image
#from PIL import Image
import torch
import torchvision
import yacs.config
import albumentations
from time import sleep


class CenterCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.CenterCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class RandomCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomCrop(
            config.dataset.image_size,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomResizeCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomResizedCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data) #return PIL image

class Resize:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.Resize(config.tta.resize)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

""" Added Albumentations """
class ShiftScaleRotate:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = albumentations.ShiftScaleRotate(p = 0.3)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']  

class RandomRotate90:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = albumentations.RandomRotate90(p = 0.3)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']

class RandomGridShuffle:
    def __init__(self, args):
        self.transform = albumentations.RandomGridShuffle(
            args.randomgridshuffle_grid,
            p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class CenterCrop:
    def __init__(self, args):
        self.transform = albumentations.CenterCrop(
            args.center_crop, args.center_crop, 
            p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class Transpose:
    def __init__(self, args):
        self.transform = albumentations.Transpose(p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class ColorJitter:
    def __init__(self, args):
        self.transform = albumentations.ColorJitter(
        p = 1  )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

"""----------------------------------------------------"""
class ToTensor:
    def __call__(
        self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))
