# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset

import flash.vision.segmentation.transforms as T
from flash.core.classification import SegmentationLabels
from flash.data.auto_dataset import AutoDataset
from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.utils.imports import _MATPLOTLIB_AVAILABLE

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class SemanticSegmentationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
    ) -> None:
        """Preprocess pipeline for semantic segmentation tasks.

        Args:
            train_transform: Dictionary with the set of transforms to apply during training.
            val_transform: Dictionary with the set of transforms to apply during validation.
            test_transform: Dictionary with the set of transforms to apply during testing.
            predict_transform: Dictionary with the set of transforms to apply during prediction.
            image_size: A tuple with the expected output image size.
        """
        train_transform, val_transform, test_transform, predict_transform = self._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform, image_size
        )
        super().__init__(train_transform, val_transform, test_transform, predict_transform)

    def _resolve_transforms(
        self,
        train_transform: Optional[Union[str, Dict]] = None,
        val_transform: Optional[Union[str, Dict]] = None,
        test_transform: Optional[Union[str, Dict]] = None,
        predict_transform: Optional[Union[str, Dict]] = None,
        image_size: Tuple[int, int] = (196, 196),
    ) -> Tuple[Dict[str, Callable], ...]:

        if not train_transform or train_transform == 'default':
            train_transform = T.default_train_transforms(image_size)

        if not val_transform or val_transform == 'default':
            val_transform = T.default_val_transforms(image_size)

        if not test_transform or test_transform == 'default':
            test_transform = T.default_val_transforms(image_size)

        if not predict_transform or predict_transform == 'default':
            predict_transform = T.default_val_transforms(image_size)

        return (
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

    def load_sample(self, sample: Union[str, Tuple[str,
                                                   str]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(sample, (
            str,
            tuple,
        )):
            raise TypeError(f"Invalid type, expected `str` or `tuple`. Got: {sample}.")

        if isinstance(sample, str):  # case for predict
            return torchvision.io.read_image(sample)

        # unpack data paths
        img_path: str = sample[0]
        img_labels_path: str = sample[1]

        # load images directly to torch tensors
        img: torch.Tensor = torchvision.io.read_image(img_path)  # CxHxW
        img_labels: torch.Tensor = torchvision.io.read_image(img_labels_path)  # CxHxW

        return {'images': img, 'masks': img_labels}

    def post_tensor_transform(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sample, torch.Tensor):  # case for predict
            out = sample.float() / 255.  # TODO: define predict transforms
            return out

        if not isinstance(sample, dict):
            raise TypeError(f"Invalid type, expected `dict`. Got: {sample}.")

        # arrange data as floating point and batch before the augmentations
        sample['images'] = sample['images'][None].float().contiguous()  # 1xCxHxW
        sample['masks'] = sample['masks'][None, :1].float().contiguous()  # 1x1xHxW

        out: Dict[str, torch.Tensor] = self.current_transform(sample)

        return out['images'][0], out['masks'][0, 0].long()

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

    @staticmethod
    def _check_valid_filepaths(filepaths: List[str]):
        if filepaths is not None and (
            not isinstance(filepaths, list) or not all(isinstance(n, str) for n in filepaths)
        ):
            raise MisconfigurationException(f"`filepaths` must be of type List[str]. Got: {filepaths}.")

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return _MatplotlibVisualization(*args, **kwargs)

    def set_labels_map(self, labels_map: Dict[int, Tuple[int, int, int]]):
        self.data_fetcher.labels_map = labels_map

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: List[str],
        train_labels: List[str],
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
        val_split: Optional[float] = None,  # MAKES IT CRASH. NEED TO BE FIXED
        **kwargs,  # TODO: remove and make explicit params
    ) -> 'SemanticSegmentationData':
        """Creates a Semantic SegmentationData object from a given list of paths to images and labels.

        Args:
            train_filepaths: List of file paths for training images.
            train_labels: List of file path for the training image labels.
            val_filepaths: List of file paths for validation images.
            val_labels: List of file path for the validation image labels.
            test_filepaths: List of file paths for testing images.
            test_labels: List of file path for the testing image labels.
            predict_filepaths: List of file paths for predicting images.
            train_transform: Image and mask transform to use for the train set.
            val_transform: Image and mask transform to use for the validation set.
            test_transform: Image and mask transform to use for the test set.
            predict_transform: Image transform to use for the predict set.
            image_size: A tuple with the expected output image size.
            batch_size: The batch size to use for parallel loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.
            data_fetcher: An optional data fetcher object instance.
            preprocess: An optional `SemanticSegmentationPreprocess` object instance.
            val_split: Float number to control the percentage of train/validation samples
                from the ``train_filepaths`` and ``train_labels`` list.


        Returns:
            SemanticSegmentationData: The constructed data module.

        """

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
        )

        # this functions overrides `DataModule.from_load_data_inputs`
        return cls.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            # predict_load_data_input=predict_filepaths,  # TODO: is it really used ?
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
    labels_map: Dict[int, Tuple[int, int, int]] = {}

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
            sample = data[i]
            if isinstance(sample, dict):
                image = sample['images']
                label = sample['masks']
            elif isinstance(sample, tuple):
                image = sample[0]
                label = sample[1]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images and labels to numpy and stack horizontally
            image_vis: np.ndarray = self._to_numpy(image.byte())
            label_tmp: torch.Tensor = SegmentationLabels.labels_to_image(label.squeeze().byte(), self.labels_map)
            label_vis: np.ndarray = self._to_numpy(label_tmp)
            img_vis = np.hstack((image_vis, label_vis))
            # send to visualiser
            ax.imshow(img_vis)
            ax.axis('off')
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_post_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)
