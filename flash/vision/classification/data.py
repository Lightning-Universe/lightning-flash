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
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T

from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.process import DefaultPreprocess
from flash.data.transforms import ApplyToKeys
from flash.utils.imports import _KORNIA_AVAILABLE, _MATPLOTLIB_AVAILABLE
from flash.vision.data import ImageFilesDataSource, ImageFoldersDataSource

if _KORNIA_AVAILABLE:
    import kornia as K

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class ImageClassificationPreprocess(DefaultPreprocess):

    data_sources = [ImageFoldersDataSource, ImageFilesDataSource]
    to_tensor = T.ToTensor()

    def __init__(
        self,
        train_transform: Optional[Union[Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Dict[str, Callable]]] = None,
        image_size: Tuple[int, int] = (196, 196),
    ):
        train_transform, val_transform, test_transform, predict_transform = self._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform, image_size
        )
        self.image_size = image_size
        super().__init__(train_transform, val_transform, test_transform, predict_transform)

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "train_transform": self._train_transform,
            "val_transform": self._val_transform,
            "test_transform": self._test_transform,
            "predict_transform": self._predict_transform,
            "image_size": self.image_size
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def default_train_transforms(self, image_size: Tuple[int, int]) -> Dict[str, Callable]:
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": nn.Sequential(
                    ApplyToKeys('input', torchvision.transforms.ToTensor()),
                    ApplyToKeys('target', torch.as_tensor),
                ),
                "post_tensor_transform": ApplyToKeys(
                    'input',
                    # TODO (Edgar): replace with resize once kornia is fixed
                    K.augmentation.RandomResizedCrop(image_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                    K.augmentation.RandomHorizontalFlip(),
                ),
                "per_batch_transform_on_device": ApplyToKeys(
                    'input',
                    K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            return {
                "pre_tensor_transform": ApplyToKeys('input', T.Resize(image_size), T.RandomHorizontalFlip()),
                "to_tensor_transform": nn.Sequential(
                    ApplyToKeys('input', torchvision.transforms.ToTensor()),
                    ApplyToKeys('target', torch.as_tensor),
                ),
                "post_tensor_transform": ApplyToKeys(
                    'input',
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ),
            }

    def default_val_transforms(self, image_size: Tuple[int, int]) -> Dict[str, Callable]:
        if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                "to_tensor_transform": nn.Sequential(
                    ApplyToKeys('input', torchvision.transforms.ToTensor()),
                    ApplyToKeys('target', torch.as_tensor),
                ),
                "post_tensor_transform": ApplyToKeys(
                    'input',
                    # TODO (Edgar): replace with resize once kornia is fixed
                    K.augmentation.RandomResizedCrop(image_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                ),
                "per_batch_transform_on_device": ApplyToKeys(
                    'input',
                    K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                )
            }
        else:
            return {
                "pre_tensor_transform": ApplyToKeys('input', T.Resize(image_size)),
                "to_tensor_transform": nn.Sequential(
                    ApplyToKeys('input', torchvision.transforms.ToTensor()),
                    ApplyToKeys('target', torch.as_tensor),
                ),
                "post_tensor_transform": ApplyToKeys(
                    'input',
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ),
            }

    def _resolve_transforms(
        self,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        image_size: Tuple[int, int] = (196, 196),
    ):

        if not train_transform or train_transform == 'default':
            train_transform = self.default_train_transforms(image_size)

        if not val_transform or val_transform == 'default':
            val_transform = self.default_val_transforms(image_size)

        if not test_transform or test_transform == 'default':
            test_transform = self.default_val_transforms(image_size)

        if not predict_transform or predict_transform == 'default':
            predict_transform = self.default_val_transforms(image_size)

        return (
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

    def collate(self, samples: Sequence[Dict[str, Any]]) -> Any:
        # todo: Kornia transforms add batch dimension which need to be removed
        for sample in samples:
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = sample[key].squeeze(0)
        return default_collate(samples)

    def common_step(self, sample: Any) -> Any:
        # if isinstance(sample, (list, tuple)):
        #     source, target = sample
        #     return self.current_transform(source), target
        return self.current_transform(sample)

    def pre_tensor_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def to_tensor_transform(self, sample: Any) -> Any:
        if self.current_transform == self._identity:
            if isinstance(sample, (list, tuple)):
                source, target = sample
                if isinstance(source, torch.Tensor):
                    return source, target
                return self.to_tensor(source), target
            elif isinstance(sample, torch.Tensor):
                return sample
            return self.to_tensor(sample)
        if isinstance(sample, torch.Tensor):
            return sample
        return self.common_step(sample)

    def post_tensor_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def per_batch_transform(self, sample: Any) -> Any:
        return self.common_step(sample)

    def per_batch_transform_on_device(self, sample: Any) -> Any:
        return self.common_step(sample)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    preprocess_cls = ImageClassificationPreprocess

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)


class MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """
    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows

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
            if isinstance(data, list):
                _img, _label = data[i]
            elif isinstance(data, tuple):
                imgs, labels = data
                _img, _label = imgs[i], labels[i]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images to numpy
            _img: np.ndarray = self._to_numpy(_img)
            if isinstance(_label, torch.Tensor):
                _label = _label.squeeze().tolist()
            # show image and set label as subplot title
            ax.imshow(_img)
            ax.set_title(str(_label))
            ax.axis('off')
        plt.show(block=self.block_viz_window)

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_pre_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_to_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        win_title: str = f"{running_stage} - show_post_tensor_transform"
        self._show_images_and_labels(samples, len(samples), win_title)

    def show_per_batch_transform(self, batch: List[Any], running_stage):
        win_title: str = f"{running_stage} - show_per_batch_transform"
        self._show_images_and_labels(batch[0], batch[0][0].shape[0], win_title)
