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
from os.path import basename, dirname, exists, isdir, isfile, join, split
from typing import Sequence

import numpy as np
import yaml
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset

from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE

if _POINTCLOUD_AVAILABLE:

    from open3d._ml3d.datasets.utils import DataProcessing
    from open3d._ml3d.utils.config import Config

    class SequencesDataset(Dataset):

        def __init__(
            self,
            data,
            cache_dir='./logs/cache',
            use_cache=False,
            num_points=65536,
            ignored_label_inds=[],
            predicting=False,
            **kwargs
        ):

            super().__init__()

            kwargs["cache_dir"] = cache_dir
            kwargs["use_cache"] = use_cache
            kwargs["num_points"] = num_points
            kwargs["ignored_label_inds"] = ignored_label_inds

            self.cfg = Config(kwargs)
            self.predicting = predicting

            if not predicting:
                self.on_fit(data)
            else:
                self.on_predict(data)

        def on_fit(self, dataset_path):
            self.split = basename(dataset_path)

            root_dir = dirname(dataset_path)
            self.dataset_path = dataset_path
            meta_file = join(root_dir, "meta.yaml")
            if not exists(meta_file):
                raise MisconfigurationException(
                    f"The {root_dir} should contain a `meta.yaml` file about the pointcloud sequences."
                )

            with open(meta_file, 'r') as f:
                self.meta = yaml.safe_load(f)

            self.label_to_names = self.get_label_to_names()
            self.num_classes = len(self.label_to_names)

            self.make_datasets()

        def make_datasets(self):
            self.path_list = []
            for seq in os.listdir(self.dataset_path):
                sequence_path = join(self.dataset_path, seq)
                directories = [f for f in os.listdir(sequence_path) if isdir(join(sequence_path, f)) and f != "labels"]
                assert len(directories) == 1
                scan_dir = join(sequence_path, directories[0])
                for scan_name in os.listdir(scan_dir):
                    self.path_list.append(join(scan_dir, scan_name))

        def on_predict(self, inputs):
            if isinstance(inputs, Sequence) and not all(isfile(p) for p in inputs):
                raise MisconfigurationException("The predict function takes only a list of paths")
            self.path_list = inputs
            self.split = "predict"

        def get_label_to_names(self):
            """Returns a label to names dictonary object.
            Returns:
                A dict where keys are label numbers and
                values are the corresponding names.
            """
            return self.meta["labels"]

        def __getitem__(self, index):
            breakpoint()
            return self.get_data(index)

        def get_data(self, idx):
            pc_path = self.path_list[idx]
            breakpoint()
            points = DataProcessing.load_pc_kitti(pc_path)

            if not self.predicting:
                dir, file = split(pc_path)
                label_path = join(dir, '../labels', file[:-4] + '.label')
                if not exists(label_path):
                    labels = np.zeros(np.shape(points)[0], dtype=np.int32)
                    if self.split not in ['test', 'all']:
                        raise FileNotFoundError(f' Label file {label_path} not found')

                else:
                    labels = DataProcessing.load_label_kitti(label_path, self.remap_lut_val).astype(np.int32)

            else:
                labels = None

            data = {
                'point': points[:, 0:3],
                'feat': None,
                'label': labels,
            }

            return data

        def get_attr(self, idx):
            pc_path = self.path_list[idx]
            dir, file = split(pc_path)
            _, seq = split(split(dir)[0])
            name = '{}_{}'.format(seq, file[:-4])

            pc_path = str(pc_path)
            attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': self.split}
            return attr

        def __len__(self):
            return len(self.path_list)
