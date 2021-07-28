import functools
import urllib.error

from pytorch_lightning.utilities import rank_zero_warn


def catch_url_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, pretrained=False, **kwargs):
        try:
            return fn(*args, pretrained=pretrained, **kwargs)
        except urllib.error.URLError:
            result = fn(*args, pretrained=False, **kwargs)
            rank_zero_warn(
                "Failed to download pretrained weights for the selected backbone. The backbone has been created with"
                " `pretrained=False` instead. If you are loading from a local checkpoint, this warning can be safely"
                " ignored.", UserWarning
            )
            return result

    return wrapper
