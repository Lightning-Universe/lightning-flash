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
from typing import Any, Type

import requests
import torch
from tqdm.auto import tqdm as tq


# Code taken from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603
# __author__  = "github.com/ruxi"
# __license__ = "MIT"
def download_file(url: str, path: str, verbose: bool = False) -> None:
    """
    Download file with progressbar

    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split('/')[-1])
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length']) if 'Content-Length' in r.headers else 0
    chunk = 1
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    if not os.path.exists(local_filename):
        with open(local_filename, 'wb') as fp:
            for chunk in tq(
                r.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit='KB',
                desc=local_filename,
                leave=True  # progressbar stays
            ):
                fp.write(chunk)  # type: ignore

    if '.zip' in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall(path)


def download_data(url: str, path: str = "data/") -> None:
    """
    Downloads data automatically from the given url to the path. Defaults to data/ for the path.
    Automatically handles .csv, .zip

    Example::

        from flash import download_data

    Args:
        url: path
        path: local

    """
    download_file(url, path)


def _contains_any_tensor(value: Any, dtype: Type = torch.Tensor) -> bool:
    # TODO: we should refactor FlashDatasetFolder to better integrate
    # with DataPipeline. That way, we wouldn't need this check.
    # This is because we are running transforms in both places.
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False
