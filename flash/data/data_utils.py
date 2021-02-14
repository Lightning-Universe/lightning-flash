import pandas as pd


def labels_from_categorical_csv(csv, index_col, return_dict=True):
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
    del df[index_col]

    # everything else is binary
    labels = df.to_numpy().argmax(1).tolist()

    if return_dict:
        labels = {name: label for name, label in zip(names, labels)}

    return labels
