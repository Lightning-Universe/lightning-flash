

def load_model(name):
    if name == 'simclr-imagenet':
        return load_simclr_imagenet()

    if name == 'swav-imagenet':
        return load_swav_imagenet()


def load_simclr_imagenet():
    from pl_bolts.models.self_supervised import SimCLR
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    model_config = {
        'model': simclr.encoder,
        'emb_size': 2048
    }
    return model_config


def load_swav_imagenet():
    from pl_bolts.models.self_supervised import SwAV

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(weight_path, strict=True)
    model_config = {
        'model': swav.model,
        'num_features': 3000
    }
    return model_config

models = {
    # 'simclr-imagenet': load_simclr_imagenet,
    'swav-imagenet': load_swav_imagenet
}
