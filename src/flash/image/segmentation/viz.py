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
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _MATPLOTLIB_AVAILABLE, Image, requires
from flash.core.utilities.stages import RunningStage
from flash.image.segmentation.output import SegmentationLabelsOutput

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None


class SegmentationMatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib."""

    def __init__(self, labels_map: Dict[int, Tuple[int, int, int]]):
        super().__init__()

        self.max_cols: int = 4  # maximum number of columns we accept
        self.block_viz_window: bool = True  # parameter to allow user to block visualisation windows
        self.labels_map: Dict[int, Tuple[int, int, int]] = labels_map

    @staticmethod
    @requires("image")
    def _to_numpy(img: Union[Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, np.ndarray):
            out = img
        elif isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    @requires("matplotlib")
    def _show_images_and_labels(
        self,
        data: List[Any],
        num_samples: int,
        title: str,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        num_samples = max(1, min(num_samples, limit_nb_samples))

        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        # create figure and set title
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            # unpack images and labels
            sample = data[i]
            if isinstance(sample, dict):
                image = sample[DataKeys.INPUT]
                label = sample[DataKeys.TARGET]
            elif isinstance(sample, tuple):
                image = sample[0]
                label = sample[1]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images and labels to numpy and stack horizontally
            image_vis: np.ndarray = self._to_numpy(image)
            label_tmp: Tensor = SegmentationLabelsOutput.labels_to_image(
                torch.as_tensor(label).squeeze(), self.labels_map
            )
            label_vis: np.ndarray = self._to_numpy(label_tmp)
            img_vis = np.hstack((image_vis, label_vis))
            # send to visualiser
            ax.imshow(img_vis)
            ax.axis("off")
        plt.show(block=self.block_viz_window)

    def show_load_sample(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title, limit_nb_samples, figsize)

    def show_per_sample_transform(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        win_title: str = f"{running_stage} - show_per_sample_transform"
        self._show_images_and_labels(samples, len(samples), win_title, limit_nb_samples, figsize)
