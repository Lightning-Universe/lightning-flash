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
from typing import Any, List, Union

import numpy as np
import torch
from pytorch_lightning.trainer.states import RunningStage

from flash.core.data.base_viz import BaseVisualization  # for viz
from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _MATPLOTLIB_AVAILABLE, _PIL_AVAILABLE, requires, requires_extras

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

if _PIL_AVAILABLE:
    from PIL import Image
else:

    class Image:
        Image = None


class MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib.
    """
    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows

    @staticmethod
    @requires_extras("image")
    def _to_numpy(img: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, torch.Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    @requires("matplotlib")
    def _show_images_and_labels(self, data: List[Any], num_samples: int, title: str):
        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        # create figure and set title
        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(title)

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in enumerate(axs):
            # unpack images and labels
            if isinstance(data, list):
                _img, _label = data[i][DefaultDataKeys.INPUT], data[i].get(DefaultDataKeys.TARGET, "")
            elif isinstance(data, dict):
                _img, _label = data[DefaultDataKeys.INPUT][i], data.get(DefaultDataKeys.TARGET, [""] * (i + 1))[i]
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
