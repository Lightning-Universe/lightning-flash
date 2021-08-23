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
from os.path import basename, dirname, exists, isdir, isfile, join
from typing import Any, Dict, List, Optional, Union

import yaml
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import BaseDataFormat, DataSource
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE

if _POINTCLOUD_AVAILABLE:
    from open3d._ml3d.datasets.kitti import DataProcessing, KITTI


class PointCloudObjectDetectionDataFormat(BaseDataFormat):
    KITTI = "kitti"


class BasePointCloudObjectDetectorLoader:

    pass


class KITTIPointCloudObjectDetectorLoader(BasePointCloudObjectDetectorLoader):
    def __init__(
        self,
        image_size: tuple = (375, 1242),
        scans_folder_name: Optional[str] = "scans",
        labels_folder_name: Optional[str] = "labels",
        calibrations_folder_name: Optional[str] = "calibs",
        **kwargs,
    ):

        self.image_size = image_size
        self.scans_folder_name = scans_folder_name
        self.labels_folder_name = labels_folder_name
        self.calibrations_folder_name = calibrations_folder_name

    def load_meta(self, root_dir, dataset: Optional[BaseAutoDataset]):
        meta_file = join(root_dir, "meta.yaml")
        if not exists(meta_file):
            raise MisconfigurationException(f"The {root_dir} should contain a `meta.yaml` file about the classes.")

        with open(meta_file) as f:
            self.meta = yaml.safe_load(f)

        if "label_to_names" not in self.meta:
            raise MisconfigurationException(
                f"The {root_dir} should contain a `meta.yaml` file about the classes with the field `label_to_names`."
            )

        dataset.num_classes = len(self.meta["label_to_names"])
        dataset.label_to_names = self.meta["label_to_names"]
        dataset.color_map = self.meta["color_map"]

    def load_data(self, folder: str, dataset: Optional[BaseAutoDataset]):
        sub_directories = os.listdir(folder)
        if len(sub_directories) != 3:
            raise MisconfigurationException(
                f"Using KITTI Format, the {folder} should contains 3 directories "
                "for ``calibrations``, ``labels`` and ``scans``."
            )

        assert self.scans_folder_name in sub_directories
        assert self.labels_folder_name in sub_directories
        assert self.calibrations_folder_name in sub_directories

        scans_dir = join(folder, self.scans_folder_name)
        labels_dir = join(folder, self.labels_folder_name)
        calibrations_dir = join(folder, self.calibrations_folder_name)

        scan_paths = [join(scans_dir, f) for f in os.listdir(scans_dir)]
        label_paths = [join(labels_dir, f) for f in os.listdir(labels_dir)]
        calibration_paths = [join(calibrations_dir, f) for f in os.listdir(calibrations_dir)]

        assert len(scan_paths) == len(label_paths) == len(calibration_paths)

        self.load_meta(dirname(folder), dataset)

        dataset.path_list = scan_paths

        return [
            {"scan_path": scan_path, "label_path": label_path, "calibration_path": calibration_path}
            for scan_path, label_path, calibration_path, in zip(scan_paths, label_paths, calibration_paths)
        ]

    def load_sample(
        self, sample: Dict[str, str], dataset: Optional[BaseAutoDataset] = None, has_label: bool = True
    ) -> Any:
        pc = KITTI.read_lidar(sample["scan_path"])
        calib = KITTI.read_calib(sample["calibration_path"])
        label = None
        if has_label:
            label = KITTI.read_label(sample["label_path"], calib)

        reduced_pc = DataProcessing.remove_outside_points(pc, calib["world_cam"], calib["cam_img"], self.image_size)

        attr = {
            "name": basename(sample["scan_path"]),
            "path": sample["scan_path"],
            "calibration_path": sample["calibration_path"],
            "label_path": sample["label_path"] if has_label else None,
            "split": "val",
        }

        data = {
            "point": reduced_pc,
            "full_point": pc,
            "feat": None,
            "calib": calib,
            "bounding_boxes": label if has_label else None,
            "attr": attr,
        }
        return data, attr

    def load_files(self, scan_paths: Union[str, List[str]], dataset: Optional[BaseAutoDataset] = None):
        if isinstance(scan_paths, str):
            scan_paths = [scan_paths]

        def clean_fn(path: str) -> str:
            return path.replace(self.scans_folder_name, self.calibrations_folder_name).replace(".bin", ".txt")

        dataset.path_list = scan_paths

        return [{"scan_path": scan_path, "calibration_path": clean_fn(scan_path)} for scan_path in scan_paths]

    def predict_load_data(self, data, dataset: Optional[BaseAutoDataset] = None):
        if (isinstance(data, str) and isfile(data)) or (isinstance(data, list) and all(isfile(p) for p in data)):
            return self.load_files(data, dataset)
        elif isinstance(data, str) and isdir(data):
            raise NotImplementedError

    def predict_load_sample(self, data, dataset: Optional[BaseAutoDataset] = None):
        data, attr = self.load_sample(data, dataset, has_label=False)
        # hack to prevent manipulation of labels
        attr["split"] = "test"
        return data, attr


class PointCloudObjectDetectorFoldersDataSource(DataSource):
    def __init__(
        self,
        data_format: Optional[BaseDataFormat] = None,
        image_size: tuple = (375, 1242),
        **loader_kwargs,
    ):
        super().__init__()

        self.loaders = {
            PointCloudObjectDetectionDataFormat.KITTI: KITTIPointCloudObjectDetectorLoader(
                **loader_kwargs, image_size=image_size
            )
        }

        self.data_format = data_format or PointCloudObjectDetectionDataFormat.KITTI
        self.loader = self.loaders[self.data_format]

    def _validate_data(self, folder: str) -> None:
        msg = f"The provided dataset for stage {self._running_stage} should be a folder. Found {folder}."
        if not isinstance(folder, str):
            raise MisconfigurationException(msg)

        if isinstance(folder, str) and not isdir(folder):
            raise MisconfigurationException(msg)

    def load_data(
        self,
        data: Any,
        dataset: Optional[BaseAutoDataset] = None,
    ) -> Any:

        self._validate_data(data)

        return self.loader.load_data(data, dataset)

    def load_sample(self, metadata: Dict[str, str], dataset: Optional[BaseAutoDataset] = None) -> Any:

        data, metadata = self.loader.load_sample(metadata, dataset)

        preprocess_fn = getattr(dataset, "preprocess_fn", None)
        if preprocess_fn:
            data = preprocess_fn(data, metadata)

        transform_fn = getattr(dataset, "transform_fn", None)
        if transform_fn:
            data = transform_fn(data, metadata)

        return {"data": data, "attr": metadata}

    def _validate_predict_data(self, data: Union[str, List[str]]) -> None:
        msg = f"The provided predict data should be a either a folder or a single/list of scan path(s). Found {data}."
        if not isinstance(data, str) and not isinstance(data, list):
            raise MisconfigurationException(msg)

        if isinstance(data, str) and (not isfile(data) or not isdir(data)):
            raise MisconfigurationException(msg)

        if isinstance(data, list) and not all(isfile(p) for p in data):
            raise MisconfigurationException(msg)

    def predict_load_data(
        self,
        data: Any,
        dataset: Optional[BaseAutoDataset] = None,
    ) -> Any:

        self._validate_predict_data(data)

        return self.loader.predict_load_data(data, dataset)

    def predict_load_sample(
        self,
        metadata: Any,
        dataset: Optional[BaseAutoDataset] = None,
    ) -> Any:

        data, metadata = self.loader.predict_load_sample(metadata, dataset)

        preprocess_fn = getattr(dataset, "preprocess_fn", None)
        if preprocess_fn:
            data = preprocess_fn(data, metadata)

        transform_fn = getattr(dataset, "transform_fn", None)
        if transform_fn:
            data = transform_fn(data, metadata)

        return {"data": data, "attr": metadata}
