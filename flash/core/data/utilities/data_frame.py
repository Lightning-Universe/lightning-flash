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
from functools import partial
from typing import Any, Callable, List, Optional, Union

import pandas as pd

from flash.core.data.utilities.paths import PATH_TYPE


def _resolve_multi_target(target_keys: List[str], row: pd.Series) -> List[Any]:
    return [row[target_key] for target_key in target_keys]


def resolve_targets(data_frame: pd.DataFrame, target_keys: Union[str, List[str]]) -> List[Any]:
    """Given a data frame and a target key or list of target keys, this function returns a list of targets.

    Args:
        data_frame: The ``pd.DataFrame`` containing the target column / columns.
        target_keys: The column in the data frame (or a list of columns) from which to resolve the target.
    """
    if not isinstance(target_keys, List):
        return data_frame[target_keys].tolist()
    return data_frame.apply(partial(_resolve_multi_target, target_keys), axis=1).tolist()


def _resolve_file(
    resolver: Callable[[PATH_TYPE, Any], PATH_TYPE], root: PATH_TYPE, input_key: str, row: pd.Series
) -> PATH_TYPE:
    return resolver(root, row[input_key])


def default_resolver(root: Optional[PATH_TYPE], file_id: Any) -> PATH_TYPE:
    file = os.path.join(root, file_id) if root is not None else file_id
    if os.path.isfile(file):
        return file
    raise ValueError(
        f"File ID `{file_id}` resolved to `{file}`, which does not exist. For use cases which involve first converting "
        f"the ID to a file you should pass a custom resolver when loading the data."
    )


def resolve_files(
    data_frame: pd.DataFrame, key: str, root: PATH_TYPE, resolver: Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]
) -> List[PATH_TYPE]:
    """Resolves a list of files from a given column in a data frame.

    Args:
        data_frame: The ``pd.DataFrame`` containing file IDs.
        key: The column in the data frame containing the file IDs.
        root: The root path to use when resolving files.
        resolver: The resolver function to use. This function should receive the root and a file ID as input and return
            the path to an existing file.
    """
    if resolver is None:
        resolver = default_resolver
    return data_frame.apply(partial(_resolve_file, resolver, root, key), axis=1).tolist()
