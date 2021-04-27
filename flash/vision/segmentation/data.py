from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import kornia as K
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.utils.imports import _KORNIA_AVAILABLE, _MATPLOTLIB_AVAILABLE

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


# container to apply augmentations at both image and mask reusing the same parameters
# TODO: we have to figure out how to decide what transforms are applied to mask
# For instance, color transforms cannot be applied to masks
class SegmentationSequential(nn.Sequential):

    def __init__(self, *args):
        super(SegmentationSequential, self).__init__(*args)

    @torch.no_grad()
    def forward(self, img, mask):
        img_out = img.float()
        mask_out = mask[None].float()
        for aug in self.children():
            img_out = aug(img_out)
            # some transforms don't have params
            if hasattr(aug, "_params"):
                mask_out = aug(mask_out, aug._params)
            else:
                mask_out = aug(mask_out)
        return img_out[0], mask_out[0, 0].long()


def to_tensor(self, x):
    return K.utils.image_to_tensor(np.array(x))


class SemanticSegmentationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        map_labels: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> 'SemanticSegmentationPreprocess':
        self._map_labels = map_labels

        # TODO: implement me
        '''train_transform, val_transform, test_transform, predict_transform = self._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform
        )'''
        augs_train = SegmentationSequential(
            K.geometry.Resize(image_size, interpolation='nearest'),
            K.augmentation.RandomHorizontalFlip(p=0.75),
        )
        augs = SegmentationSequential(
            K.geometry.Resize(image_size, interpolation='nearest'),
            K.augmentation.RandomHorizontalFlip(p=0.),
        )
        augs_pred = nn.Sequential(K.geometry.Resize(image_size, interpolation='nearest'), )
        train_transform = dict(to_tensor_transform=augs_train)
        val_transform = dict(to_tensor_transform=augs)
        test_transform = dict(to_tensor_transform=augs)
        predict_transform = dict(to_tensor_transform=augs_pred)

        super().__init__(train_transform, val_transform, test_transform, predict_transform)

    def _image_to_labels(self, img) -> torch.Tensor:
        assert len(img.shape) == 3, img.shape
        C, H, W = img.shape
        outs = torch.empty(H, W, dtype=torch.int64)
        for label, values in self._map_labels.items():
            vals = torch.tensor(values).view(3, 1, 1)
            mask = (img == vals).all(-3)
            outs[mask] = label
        return outs

    # TODO: is it a problem to load sample directly in tensor. What happens in to_tensor_tranform
    def load_sample(self, sample: Union[str, Tuple[str,
                                                   str]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(sample, (
            str,
            tuple,
        )):
            raise TypeError(f"Invalid type, expected `tuple`. Got: {sample}.")

        if isinstance(sample, str):  # case for predict
            return torchvision.io.read_image(sample)

        # unpack data paths
        img_path: str = sample[0]
        img_labels_path: str = sample[1]

        # load images directly to torch tensors
        img: torch.Tensor = torchvision.io.read_image(img_path)  # CxHxW
        img_labels: torch.Tensor = torchvision.io.read_image(img_labels_path)  # CxHxW
        # TODO: need to figure best api for this
        img_labels = img_labels[0]  # HxW

        return img, img_labels

    def to_tensor_transform(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sample, torch.Tensor):  # case for predict
            out = sample.float() / 255.  # TODO: define predict transforms
            return out

        if not isinstance(sample, tuple):
            raise TypeError(f"Invalid type, expected `tuple`. Got: {sample}.")
        img, img_labels = sample
        img_out, img_labels_out = self.current_transform(img, img_labels)

        # TODO: decide at which point do we apply this
        if self._map_labels is not None:
            img_labels_out = self._image_to_labels(img_labels_out)

        return img_out, img_labels_out

    # TODO: the labels are not clear how to forward to the loss once are transform from this point
    '''def per_batch_transform(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(sample, list):
            raise TypeError(f"Invalid type, expected `tuple`. Got: {sample}.")
        img, img_labels = sample
        # THIS IS CRASHING
        # out1 = self.current_transform(img)  # images
        # out2 = self.current_transform(img_labels)  # labels
        # return out1, out2
        return img, img_labels

    # TODO: the labels are not clear how to forward to the loss once are transform from this point
    def per_batch_transform_on_device(self, sample: Any) -> Any:
        pass'''


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    # TODO: figure out if this needed
    #def __init__(self, **kwargs) -> None:
    #    super().__init__(**kwargs)

    @staticmethod
    def _check_valid_filepaths(filepaths: List[str]):
        if filepaths is not None and (
            not isinstance(filepaths, list) or not all(isinstance(n, str) for n in filepaths)
        ):
            raise MisconfigurationException(f"`filepaths` must be of type List[str]. Got: {filepaths}.")

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return _MatplotlibVisualization(*args, **kwargs)

    def set_map_labels(self, map_labels):
        self.data_fetcher.map_labels = map_labels

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[List[str]] = None,
        train_labels: Optional[List[str]] = None,
        val_filepaths: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        test_filepaths: Optional[List[str]] = None,
        test_labels: Optional[List[str]] = None,
        predict_filepaths: Optional[List[str]] = None,
        train_transform: Union[str, Dict] = 'default',
        val_transform: Union[str, Dict] = 'default',
        test_transform: Union[str, Dict] = 'default',
        predict_transform: Union[str, Dict] = 'default',
        image_size: Tuple[int, int] = (196, 196),
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        #seed: Optional[int] = 42,  # SEED NEVER USED
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        # val_split: Optional[float] = None,  # MAKES IT CRASH. NEED TO BE FIXED
        map_labels: Optional[Dict[int, Tuple[int, int, int]]] = None,
        **kwargs,  # TODO: remove and make explicit params
    ) -> 'SemanticSegmentationData':

        # verify input data format
        SemanticSegmentationData._check_valid_filepaths(train_filepaths)
        SemanticSegmentationData._check_valid_filepaths(train_labels)
        SemanticSegmentationData._check_valid_filepaths(val_filepaths)
        SemanticSegmentationData._check_valid_filepaths(val_labels)
        SemanticSegmentationData._check_valid_filepaths(test_filepaths)
        SemanticSegmentationData._check_valid_filepaths(test_labels)
        SemanticSegmentationData._check_valid_filepaths(predict_filepaths)

        # create the preprocess objects
        preprocess = preprocess or SemanticSegmentationPreprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            image_size=image_size,
            map_labels=map_labels,
        )

        # this functions overrides `DataModule.from_load_data_inputs`
        return cls.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            predict_load_data_input=predict_filepaths,
            batch_size=batch_size,
            num_workers=num_workers,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            #seed=seed, # THIS CRASHES
            #val_split=val_split,  # THIS CRASHES
            **kwargs,  # TODO: remove and make explicit params
        )


class _MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """
    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows
    map_labels = {}

    @staticmethod
    def _to_numpy(img: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, torch.Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    def _labels_to_image(self, img_labels: torch.Tensor) -> torch.Tensor:
        assert len(img_labels.shape) == 2, img_labels.shape
        H, W = img_labels.shape
        out = torch.empty(3, H, W, dtype=torch.uint8)
        for label_id, label_val in self.map_labels.items():
            mask = (img_labels == label_id)
            for i in range(3):
                out[i].masked_fill_(mask, label_val[i])
        return out

    def _show_images_and_labels(self, data: List[Any], num_samples: int, title: str):
        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        if not _MATPLOTLIB_AVAILABLE:
            raise MisconfigurationException("You need matplotlib to visualise. Please, pip install matplotlib")

        # create figure and set title
        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(title)

        for i, ax in enumerate(axs.ravel()):
            # unpack images and labels
            if isinstance(data, list):
                _img, _img_labels = data[i]
            elif isinstance(data, tuple):
                imgs, imgs_labels = data
                _img, _img_labels = imgs[i], imgs_labels[i]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images and labels to numpy and stack horizontally
            img_vis: np.ndarray = self._to_numpy(_img.byte())
            _img_labels = self._labels_to_image(_img_labels.byte())
            img_labels_vis: np.ndarray = self._to_numpy(_img_labels)
            img_vis = np.hstack((img_vis, img_labels_vis))
            # send to visualiser
            ax.imshow(img_vis)
            ax.axis('off')
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_to_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)
