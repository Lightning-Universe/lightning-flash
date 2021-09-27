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

import os.path
import zipfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Type

import requests
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch import Tensor
from tqdm.auto import tqdm as tq

_STAGES_PREFIX = {
    RunningStage.TRAINING: "train",
    RunningStage.TESTING: "test",
    RunningStage.VALIDATING: "val",
    RunningStage.PREDICTING: "predict",
}
_STAGES_PREFIX_VALUES = {"train", "test", "val", "predict"}

_DATASOURCE_FUNCS: Set[str] = {
    "load_data",
    "load_sample",
}

_PREPROCESS_FUNCS: Set[str] = {
    "pre_tensor_transform",
    "to_tensor_transform",
    "post_tensor_transform",
    "per_batch_transform",
    "per_sample_transform_on_device",
    "per_batch_transform_on_device",
    "collate",
}

_CALLBACK_FUNCS: Set[str] = {
    "load_sample",
    *_PREPROCESS_FUNCS,
}

_POSTPROCESS_FUNCS: Set[str] = {
    "per_batch_transform",
    "uncollate",
    "per_sample_transform",
    "save_sample",
    "save_data",
}


class CurrentRunningStageContext:
    def __init__(self, running_stage: RunningStage, obj: Any, reset: bool = True):
        self._running_stage = running_stage
        self._obj = obj
        self._reset = reset

    def __enter__(self):
        if self._obj is not None:
            if getattr(self._obj, "running_stage", None) != self._running_stage:
                self._obj.running_stage = self._running_stage
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._obj is not None and self._reset:
            self._obj.running_stage = None


class CurrentFuncContext:
    def __init__(self, current_fn: str, obj: Any):
        self._current_fn = current_fn
        self._obj = obj

    def __enter__(self):
        if self._obj is not None:
            if getattr(self._obj, "current_fn", None) != self._current_fn:
                self._obj.current_fn = self._current_fn
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._obj is not None:
            self._obj.current_fn = None


class CurrentRunningStageFuncContext:
    def __init__(self, running_stage: RunningStage, current_fn: str, obj: Any):
        self._running_stage = running_stage
        self._current_fn = current_fn
        self._obj = obj

    def __enter__(self):
        if self._obj is not None:
            if getattr(self._obj, "running_stage", None) != self._running_stage:
                self._obj.running_stage = self._running_stage
            if getattr(self._obj, "current_fn", None) != self._current_fn:
                self._obj.current_fn = self._current_fn
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._obj is not None:
            self._obj.running_stage = None
            self._obj.current_fn = None


def download_data(url: str, path: str = "data/", verbose: bool = False) -> None:
    """Download file with progressbar.

    # Code taken from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603
    # __author__  = "github.com/ruxi"
    # __license__ = "MIT"

    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if url == "NEED_TO_BE_CREATED":
        raise NotImplementedError

    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split("/")[-1])
    r = requests.get(url, stream=True, verify=False)
    file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    if not os.path.exists(local_filename):
        with open(local_filename, "wb") as fp:
            for chunk in tq(
                r.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit="KB",
                desc=local_filename,
                leave=True,  # progressbar stays
            ):
                fp.write(chunk)  # type: ignore

    if ".zip" in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(path)


def _contains_any_tensor(value: Any, dtype: Type = Tensor) -> bool:
    # TODO: we should refactor FlashDatasetFolder to better integrate
    # with DataPipeline. That way, we wouldn't need this check.
    # This is because we are running transforms in both places.
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value)
    if isinstance(value, dict):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False


class FuncModule(torch.nn.Module):
    """This class is used to wrap a callable within a nn.Module and apply the wrapped function in `__call__`"""

    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.func.__name__})"


def convert_to_modules(transforms: Optional[Dict[str, Callable]]):

    if transforms is None or isinstance(transforms, torch.nn.Module):
        return transforms

    transforms = apply_to_collection(transforms, Callable, FuncModule, wrong_dtype=torch.nn.Module)
    transforms = apply_to_collection(transforms, Mapping, torch.nn.ModuleDict, wrong_dtype=torch.nn.ModuleDict)
    transforms = apply_to_collection(
        transforms, Iterable, torch.nn.ModuleList, wrong_dtype=(torch.nn.ModuleList, torch.nn.ModuleDict)
    )
    return transforms
