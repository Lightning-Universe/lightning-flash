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
from typing import Any, Dict, List, Union

from flash.core.utilities.imports import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd


def labels_from_categorical_csv(
    csv: str,
    index_col: str,
    feature_cols: List,
    return_dict: bool = True,
    index_col_collate_fn: Any = None
) -> Union[Dict, List]:
    """
    Returns a dictionary with {index_col: label} for each entry in the csv.

    Expects a csv of this form:

    index_col,    b,     c,     d
    some_name,    0      0      1
    some_name_b,  1      0      0

    """
    if not _PANDAS_AVAILABLE:
        raise ModuleNotFoundError("Please, `pip install pandas`")

    df = pd.read_csv(csv)
    # get names
    names = df[index_col].to_list()

    # apply colate fn to index_col
    if index_col_collate_fn:
        for i in range(len(names)):
            names[i] = index_col_collate_fn(names[i])

    # everything else is binary
    feature_df = df[feature_cols]
    labels = feature_df.to_numpy().argmax(1).tolist()

    if return_dict:
        labels = {name: label for name, label in zip(names, labels)}

    return labels
