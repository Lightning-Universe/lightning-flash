from typing import Any, Dict, List, Union

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
