import torch

from flash.image.backbones import IMAGE_CLASSIFIER_BACKBONES


# Paper: Emerging Properties in Self-Supervised Vision Transformers
# https://arxiv.org/abs/2104.14294 from Mathilde Caron and al. (29 Apr 2021)
# weights from https://github.com/facebookresearch/dino
def dino_deits16(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_deits16')
    return backbone, 384


def dino_deits8(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_deits8')
    return backbone, 384


def dino_vitb16(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    return backbone, 768


def dino_vitb8(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    return backbone, 768


IMAGE_CLASSIFIER_BACKBONES(dino_deits16)
IMAGE_CLASSIFIER_BACKBONES(dino_deits8)
IMAGE_CLASSIFIER_BACKBONES(dino_vitb16)
IMAGE_CLASSIFIER_BACKBONES(dino_vitb8)
