from pytorch_lightning.utilities import _BOLTS_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _BOLTS_AVAILABLE:
    from pl_bolts.models.self_supervised import SimCLR, SwAV

DEFAULT_URLS = {
    "SimCLR": 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt',
    "SwAV": 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar',
}


def load_model(name):
    if name == 'simclr-imagenet':
        return load_simclr_imagenet()

    elif name == 'swav-imagenet':
        return load_swav_imagenet()

    else:
        raise MisconfigurationException("Currently, only `simclr-imagenet` and `swav-imagenet` are supported.")


def load_simclr_imagenet():
    simclr = SimCLR.load_from_checkpoint(DEFAULT_URLS["SimCLR"], strict=False)

    model_config = {'model': simclr.encoder, 'emb_size': 2048}
    return model_config


def load_swav_imagenet():
    swav = SwAV.load_from_checkpoint(DEFAULT_URLS["SwAV"], strict=True)
    model_config = {'model': swav.model, 'num_features': 3000}
    return model_config


models = {'simclr-imagenet': load_simclr_imagenet, 'swav-imagenet': load_swav_imagenet}
