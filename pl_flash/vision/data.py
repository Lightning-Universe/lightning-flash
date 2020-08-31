from typing import Union
import pathlib
import pickle

import torch


class VisionData(object):
    @staticmethod
    def load_file(path: Union[str, pathlib.Path], **kwargs):
        path = str(path)
        if path.endswith(".pt"):
            loaded = VisionData.load_torch(path, **kwargs)

        elif any([path.endswith(ext) for ext in [".npy", ".npz", ".txt"]]):
            loaded = VisionData.load_numpy(path, **kwargs)

        elif path.endswith(".pkl"):
            loaded = VisionData.load_pickle(path, **kwargs)

        else:
            loaded = VisionData.load_pillow(path, **kwargs)

        return torch.utils.data._utils.collate.default_convert(loaded)

    @staticmethod
    def load_torch(path, **kwargs):
        return torch.load(path, **kwargs)

    @staticmethod
    def load_numpy(path, **kwargs):
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is not available. Please install it with `pip install numpy`")

        if path.endswith(".npy") or path.endswith(".npz"):
            loaded = np.load(path, **kwargs)

        elif path.endswith(".txt"):
            loaded = np.loadtxt(path, **kwargs)

        else:
            raise ValueError

        return torch.from_numpy(loaded)

    @staticmethod
    def load_pickle(path, **kwargs):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pillow(path, **kwargs):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is not available. Please install it with `pip install pillow`")

        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is not available. Please install it with `pip install numpy`")

        loaded = Image.open(path, **kwargs)

        return torch.from_numpy(np.array(loaded))
