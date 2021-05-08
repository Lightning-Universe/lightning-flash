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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS

from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.data_source import DefaultDataKeys, DefaultDataSources, PathsDataSource
from flash.data.process import Preprocess
from flash.utils.imports import _MATPLOTLIB_AVAILABLE
from flash.vision.segmentation.serialization import SegmentationLabels
from flash.vision.segmentation.transforms import default_train_transforms, default_val_transforms

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class SemanticSegmentationPathsDataSource(PathsDataSource):

    def __init__(self):
        super().__init__(IMG_EXTENSIONS)

    def load_data(self, data: Union[Tuple[str, str], Tuple[List[str], List[str]]]) -> Sequence[Mapping[str, Any]]:
        input_data, target_data = data

        if self.isdir(input_data) and self.isdir(target_data):
            files = os.listdir(input_data)
            input_files = [os.path.join(input_data, file) for file in files]
            target_files = [os.path.join(target_data, file) for file in files]

            target_files = list(filter(os.path.isfile, target_files))

            if len(input_files) != len(target_files):
                rank_zero_warn(
                    f"Found inconsistent files in input_dir: {input_data} and target_dir: {target_data}. "
                    f"The following files have been dropped: "
                    f"{list(set(input_files).difference(set(target_files)))}",
                    UserWarning,
                )

            input_data = input_files
            target_data = target_files

        if not isinstance(input_data, list) and not isinstance(target_data, list):
            input_data = [input_data]
            target_data = [target_data]

        data = filter(
            lambda sample: (
                has_file_allowed_extension(sample[0], self.extensions) and
                has_file_allowed_extension(sample[1], self.extensions)
            ),
            zip(input_data, target_data),
        )

        return [{DefaultDataKeys.INPUT: input, DefaultDataKeys.TARGET: target} for input, target in data]

    def predict_load_data(self, data: Union[str, List[str]]):
        return super().predict_load_data(data)

    def load_sample(self, sample: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        # unpack data paths
        img_path = sample[DefaultDataKeys.INPUT]
        img_labels_path = sample[DefaultDataKeys.TARGET]

        # load images directly to torch tensors
        img: torch.Tensor = torchvision.io.read_image(img_path)  # CxHxW
        img_labels: torch.Tensor = torchvision.io.read_image(img_labels_path)  # CxHxW
        img_labels = img_labels[0]  # HxW

        return {DefaultDataKeys.INPUT: img.float(), DefaultDataKeys.TARGET: img_labels.float()}

    def predict_load_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        return {DefaultDataKeys.INPUT: torchvision.io.read_image(sample[DefaultDataKeys.INPUT]).float()}


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
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={DefaultDataSources.PATHS: SemanticSegmentationPathsDataSource()},
            default_data_source=DefaultDataSources.PATHS,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "image_size": self.image_size,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def collate(self, samples: Sequence[Dict[str, Any]]) -> Any:
        # todo: Kornia transforms add batch dimension which need to be removed
        for sample in samples:
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = sample[key].squeeze(0)
        return super().collate(samples)

    @property
    def default_train_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_train_transforms(self.image_size)

    @property
    def default_val_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)

    @property
    def default_test_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)

    @property
    def default_predict_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_val_transforms(self.image_size)


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    preprocess_cls = SemanticSegmentationPreprocess

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return SegmentationMatplotlibVisualization(*args, **kwargs)

    def set_labels_map(self, labels_map: Dict[int, Tuple[int, int, int]]):
        self.data_fetcher.labels_map = labels_map

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        train_target_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        return cls.from_data_source(
            DefaultDataSources.PATHS,
            (train_folder, train_target_folder),
            (val_folder, val_target_folder),
            (test_folder, test_target_folder),
            predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs,
        )


class SegmentationMatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """

    def __init__(self):
        super().__init__(self)
        self.max_cols: int = 4  # maximum number of columns we accept
        self.block_viz_window: bool = True  # parameter to allow user to block visualisation windows
        self.labels_map: Dict[int, Tuple[int, int, int]] = {}

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
                image = sample[DefaultDataKeys.INPUT]
                label = sample[DefaultDataKeys.TARGET]
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
