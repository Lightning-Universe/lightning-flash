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
import tarfile
import zipfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set

import requests
import urllib3
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch import nn
from tqdm.auto import tqdm as tq

from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage

# Skip doctests if requirements aren't available
if not _CORE_TESTING:
    __doctest_skip__ = ["download_data"]

_STAGES_PREFIX = {
    RunningStage.TRAINING: "train",
    RunningStage.TESTING: "test",
    RunningStage.VALIDATING: "val",
    RunningStage.PREDICTING: "predict",
    RunningStage.SERVING: "serve",
    RunningStage.SANITY_CHECKING: "val",
}

_INPUT_TRANSFORM_FUNCS: Set[str] = {
    "per_sample_transform",
    "per_batch_transform",
    "per_sample_transform_on_device",
    "per_batch_transform_on_device",
    "collate",
}

_CALLBACK_FUNCS: Set[str] = {
    "load_sample",
    *_INPUT_TRANSFORM_FUNCS,
}

_OUTPUT_TRANSFORM_FUNCS: Set[str] = {
    "per_batch_transform",
    "uncollate",
    "per_sample_transform",
}


def download_data(url: str, path: str = "data/", verbose: bool = False) -> None:
    """Download file with progressbar.

    # Code adapted from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603
    # __author__  = "github.com/ruxi"
    # __license__ = "MIT"

    Examples
    ________

    .. doctest::

        >>> import os
        >>> from flash.core.data.utils import download_data
        >>> download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "./data")
        >>> os.listdir("./data")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [...]
    """
    # Disable warning about making an insecure request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

    def extract_tarfile(file_path: str, extract_path: str, mode: str):
        if os.path.exists(file_path):
            with tarfile.open(file_path, mode=mode) as tar_ref:
                for member in tar_ref.getmembers():
                    try:
                        tar_ref.extract(member, path=extract_path, set_attrs=False)
                    except PermissionError:
                        raise PermissionError(f"Could not extract tar file {file_path}")

    if ".zip" in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(path)
    elif local_filename.endswith(".tar.gz") or local_filename.endswith(".tgz"):
        extract_tarfile(local_filename, path, "r:gz")
    elif local_filename.endswith(".tar.bz2") or local_filename.endswith(".tbz"):
        extract_tarfile(local_filename, path, "r:bz2")


class FuncModule(nn.Module):
    """This class is used to wrap a callable within a nn.Module and apply the wrapped function in `__call__`"""

    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.func.__name__})"

    def __repr__(self):
        return str(self.func)


def convert_to_modules(transforms: Optional[Dict[str, Callable]]):

    if transforms is None or isinstance(transforms, nn.Module):
        return transforms

    transforms = apply_to_collection(transforms, Callable, FuncModule, wrong_dtype=nn.Module)
    transforms = apply_to_collection(transforms, Mapping, nn.ModuleDict, wrong_dtype=nn.ModuleDict)
    transforms = apply_to_collection(transforms, Iterable, nn.ModuleList, wrong_dtype=(nn.ModuleList, nn.ModuleDict))
    return transforms
