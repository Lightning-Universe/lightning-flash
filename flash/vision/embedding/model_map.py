from contextlib import suppress
from typing import Type

from pytorch_lightning.utilities import _BOLTS_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _BOLTS_AVAILABLE:
    with suppress(TypeError):
        from pl_bolts.models.self_supervised import SimCLR, SwAV

ROOT_S3_BUCKET = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"


def load_simclr_imagenet(path_or_url: str = f"{ROOT_S3_BUCKET}/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"):
    simclr = SimCLR.load_from_checkpoint(path_or_url, strict=False)
    model_config = {'model': simclr.encoder, 'emb_size': 2048}
    return model_config


def load_swav_imagenet(path_or_url: str = f"{ROOT_S3_BUCKET}/swav/swav_imagenet/swav_imagenet.pth.tar"):
    swav = SwAV.load_from_checkpoint(path_or_url, strict=True)
    model_config = {'model': swav.model, 'num_features': 3000}
    return model_config


_models = {'simclr-imagenet': load_simclr_imagenet, 'swav-imagenet': load_swav_imagenet}


def _load_model(name):
    if not _BOLTS_AVAILABLE:
        raise MisconfigurationException("Bolts isn't installed. Please, use ``pip install lightning-bolts``.")
    if name in _models:
        return _models[name]()
    raise MisconfigurationException("Currently, only `simclr-imagenet` and `swav-imagenet` are supported.")
