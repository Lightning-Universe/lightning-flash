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
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union

import torch

from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat, NumpyInput, PathsInput, TensorInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.utils import image_default_loader
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.integrations.icevision.transforms import default_transforms
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _KORNIA_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.image.data import ImageDeserializer, IMG_EXTENSIONS

if _ICEVISION_AVAILABLE:
    import torchvision.transforms.functional as FT
    from icevision.parsers import COCOMaskParser, Parser, VOCMaskParser
    from torchvision.datasets.folder import has_file_allowed_extension

else:
    COCOMaskParser = object
    VOCMaskParser = object
    Parser = object
if _KORNIA_AVAILABLE:
    import kornia as K


class InstanceSegmentationNumpyInput(NumpyInput):
    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        breakpoint()
        img = torch.from_numpy(sample[DataKeys.INPUT]).float()
        sample[DataKeys.INPUT] = img
        sample[DataKeys.METADATA] = {"size": img.shape}
        return sample


class InstanceSegmentationPathsInput(PathsInput):
    def __init__(self):
        breakpoint()
        super().__init__(IMG_EXTENSIONS)

    def load_data(
        self, data: Union[Tuple[str, str], Tuple[List[str], List[str]]], dataset: BaseAutoDataset
    ) -> Sequence[Mapping[str, Any]]:
        input_data, target_data = data

        if self.isdir(input_data) and self.isdir(target_data):
            input_files = os.listdir(input_data)
            target_files = os.listdir(target_data)

            all_files = set(input_files).intersection(set(target_files))

            if len(all_files) != len(input_files) or len(all_files) != len(target_files):
                rank_zero_warn(
                    f"Found inconsistent files in input_dir: {input_data} and target_dir: {target_data}. Some files"
                    " have been dropped.",
                    UserWarning,
                )

            input_data = [os.path.join(input_data, file) for file in all_files]
            target_data = [os.path.join(target_data, file) for file in all_files]

        if not isinstance(input_data, list) and not isinstance(target_data, list):
            input_data = [input_data]
            target_data = [target_data]

        if len(input_data) != len(target_data):
            raise MisconfigurationException(
                f"The number of input files ({len(input_data)}) and number of target files ({len(target_data)}) must be"
                " the same.",
            )

        data = filter(
            lambda sample: (
                has_file_allowed_extension(sample[0], self.extensions)
                and has_file_allowed_extension(sample[1], self.extensions)
            ),
            zip(input_data, target_data),
        )

        data = [{DataKeys.INPUT: input, DataKeys.TARGET: target} for input, target in data]

        return data

    def predict_load_data(self, data: Union[str, List[str]]):
        return super().predict_load_data(data)

    def load_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Union[torch.Tensor, torch.Size]]:
        # unpack data paths
        img_path = sample[DataKeys.INPUT]
        img_labels_path = sample[DataKeys.TARGET]

        # load images directly to torch tensors
        img: torch.Tensor = FT.to_tensor(image_default_loader(img_path))  # CxHxW
        img_labels: torch.Tensor = torchvision.io.read_image(img_labels_path)  # CxHxW
        img_labels = img_labels[0]  # HxW

        sample[DataKeys.INPUT] = img.float()
        sample[DataKeys.TARGET] = img_labels.float()
        sample[DataKeys.METADATA] = {
            "filepath": img_path,
            "size": img.shape,
        }
        return sample

    @staticmethod
    def predict_load_sample(sample: Mapping[str, Any]) -> Mapping[str, Any]:

        img_path = sample[DataKeys.INPUT]
        img = FT.to_tensor(image_default_loader(img_path)).float()
        breakpoint()
        sample[DataKeys.INPUT] = img
        sample[DataKeys.METADATA] = {
            "filepath": img_path,
            "size": img.shape,
        }
        return sample


class InstanceSegmentationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
        parser: Optional[Callable] = None,
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                "coco": partial(IceVisionInput, parser=COCOMaskParser),
                "voc": partial(IceVisionInput, parser=VOCMaskParser),
                InputFormat.NUMPY: InstanceSegmentationNumpyInput(),
                InputFormat.FILES: IceVisionInput,
                InputFormat.FOLDERS: partial(IceVisionInput, parser=parser),
            },
            default_input=InputFormat.FILES,
        )

        self._default_collate = self._identity

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)


class InstanceSegmentationOutputTransform(OutputTransform):
    # @staticmethod
    # def uncollate(batch: Any) -> Any:
    #     breakpoint()
    #     return batch[DataKeys.PREDS]
    def per_sample_transform(self, sample: Any) -> Any:
        # TODO GET HERE WITH image SIZE
        breakpoint()
        resize = K.geometry.Resize(sample[DataKeys.METADATA]["original_size"][-2:], interpolation="bilinear")
        sample[DataKeys.PREDS]["mask_array"] = (
            resize(torch.tensor(sample[DataKeys.PREDS]["mask_array"], dtype=torch.float)) > 0
        ).to(torch.uint8)
        # sample[DataKeys.INPUT] = resize(sample[DataKeys.INPUT])
        return super().per_sample_transform(sample)


class InstanceSegmentationData(DataModule):

    input_transform_cls = InstanceSegmentationInputTransform

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
        parser: Optional[Union[Callable, Type[Parser]]] = None,
        **data_module_kwargs,
    ) -> "InstanceSegmentationData":
        return cls(
            IceVisionInput(RunningStage.TRAINING, train_folder, train_ann_file, parser=parser),
            IceVisionInput(RunningStage.VALIDATING, val_folder, val_ann_file, parser=parser),
            IceVisionInput(RunningStage.TESTING, test_folder, test_ann_file, parser=parser),
            IceVisionInput(RunningStage.PREDICTING, predict_folder, parser=parser),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData` object from the
        given data folders and annotation files in the COCO format.

        Args:
            train_folder: The folder containing the train data.
            train_ann_file: The COCO format annotation file.
            val_folder: The folder containing the validation data.
            val_ann_file: The COCO format annotation file.
            test_folder: The folder containing the test data.
            test_ann_file: The COCO format annotation file.
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            image_size: The size to resize images (and their masks) to.
        """
        return cls.from_folders(
            train_folder=train_folder,
            train_ann_file=train_ann_file,
            val_folder=val_folder,
            val_ann_file=val_ann_file,
            test_folder=test_folder,
            test_ann_file=test_ann_file,
            predict_folder=predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            image_size=image_size,
            parser=COCOMaskParser,
            **data_module_kwargs,
        )

    @classmethod
    def from_voc(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData` object from the
        given data folders and annotation files in the VOC format.

        Args:
            train_folder: The folder containing the train data.
            train_ann_file: The COCO format annotation file.
            val_folder: The folder containing the validation data.
            val_ann_file: The COCO format annotation file.
            test_folder: The folder containing the test data.
            test_ann_file: The COCO format annotation file.
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            image_size: The size to resize images (and their masks) to.
        """
        return cls.from_folders(
            train_folder=train_folder,
            train_ann_file=train_ann_file,
            val_folder=val_folder,
            val_ann_file=val_ann_file,
            test_folder=test_folder,
            test_ann_file=test_ann_file,
            predict_folder=predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            image_size=image_size,
            parser=VOCMaskParser,
            **data_module_kwargs,
        )
