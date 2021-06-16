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
import base64
import glob
import os
from functools import partial
from io import BytesIO
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.base_viz import BaseVisualization  # for viz
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.image.classification.transforms import default_transforms, train_default_transforms
from flash.image.data import ImageNumpyDataSource, ImagePathsDataSource, ImageTensorDataSource

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.datasets.folder import default_loader

if _IMAGE_AVAILABLE:
    from PIL import Image
    from PIL import Image as PILImage
else:

    class Image:
        Image = None


class ImageClassificationDataFrameDataSource(
    DataSource[Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str]]]
):

    @staticmethod
    def _resolve_file(root: str, input_key: str, row: pd.Series) -> pd.Series:
        file_id = row[input_key]
        if os.path.isabs(file_id):
            pattern = f"{file_id}*"
        else:
            pattern = os.path.join(root, f"*{file_id}*")
        files = glob.glob(pattern)
        if len(files) > 1:
            raise ValueError(
                f"Found multiple matches for pattern: {pattern}. Values in the {input_key} column should uniquely "
                "identify the file to load."
            )
        elif len(files) == 0:
            raise ValueError(
                f"Found no matches for pattern: {pattern}. Values in the {input_key} column should uniquely identify "
                "the file to load."
            )
        row[input_key] = files[0]
        return row

    @staticmethod
    def _resolve_multi_target(target_keys: List[str], row: pd.Series) -> pd.Series:
        row[target_keys[0]] = [row[target_key] for target_key in target_keys]
        return row

    def load_data(
        self,
        data: Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        data_frame, input_key, target_keys, root = data
        if root is None:
            root = ""
        data_frame = data_frame.apply(partial(self._resolve_file, root, input_key), axis=1)
        if not self.predicting:
            if isinstance(target_keys, List):
                data_frame = data_frame.apply(partial(self._resolve_multi_target, target_keys), axis=1)
                target_keys = target_keys[0]
            return [{
                DefaultDataKeys.INPUT: row[input_key],
                DefaultDataKeys.TARGET: row[target_keys],
            } for row in data_frame.iterrows()]
        else:
            return [{DefaultDataKeys.INPUT: row[input_key]} for row in data_frame.iterrows()]

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        sample[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return sample


class ImageClassificationCSVDataSource(ImageClassificationDataFrameDataSource):

    def load_data(
        self,
        data: Tuple[str, str, Union[str, List[str]], Optional[str]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        csv_file, input_key, target_keys, root = data
        data_frame = pd.read_csv(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data((data_frame, input_key, target_keys, root), dataset)


class ImageClassificationDeserializer(Deserializer):

    def __init__(self):

        self.to_tensor = torchvision.transforms.ToTensor()

    def deserialize(self, data: str) -> Dict:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        return {
            DefaultDataKeys.INPUT: img,
        }


class ImageClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        deserializer: Optional[Deserializer] = None,
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: ImagePathsDataSource(),
                DefaultDataSources.FOLDERS: ImagePathsDataSource(),
                DefaultDataSources.NUMPY: ImageNumpyDataSource(),
                DefaultDataSources.TENSORS: ImageTensorDataSource(),
                "data_frame": ImageClassificationDataFrameDataSource(),
                DefaultDataSources.CSV: ImageClassificationCSVDataSource(),
            },
            deserializer=deserializer or ImageClassificationDeserializer(),
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "image_size": self.image_size}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.image_size)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    preprocess_cls = ImageClassificationPreprocess

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_root: Optional[str] = None,
        val_root: Optional[str] = None,
        test_root: Optional[str] = None,
        predict_root: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        """Creates a :class:`~flash.image.classification.data.ImageClassificationData` object from the given CSV files
        using the :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.CSV` from the passed or constructed
        :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields (columns) in the CSV file to use for the input.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            train_root: The root directory to look for train image files in.
            val_root: The root directory to look for validation image files in.
            test_root: The root directory to look for test image files in.
            predict_root: The root directory to look for predict image files in.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_csv(
                "image_id",
                "target",
                train_file="train_data.csv",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, input_fields, target_fields, train_root),
            (val_file, input_fields, target_fields, val_root),
            (test_file, input_fields, target_fields, test_root),
            (predict_file, input_fields, target_fields, predict_root),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )


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
                _img, _label = data[i][DefaultDataKeys.INPUT], data[i][DefaultDataKeys.TARGET]
            elif isinstance(data, dict):
                _img, _label = data[DefaultDataKeys.INPUT][i], data[DefaultDataKeys.TARGET][i]
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
        self._show_images_and_labels(batch[0], batch[0][DefaultDataKeys.INPUT].shape[0], win_title)
