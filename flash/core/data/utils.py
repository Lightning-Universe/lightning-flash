from io import BytesIO
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

import torch


def download_data(url, path="data/"):
    """
    Downloads data automatically from the given url to the path. Defaults to data/ for the path.
    Automatically handles:

        - .csv
        - .zip

    Args:
        url: path
        path: local

    """

    if ".zip" in url:
        download_zip_data(url, path)

    else:
        download_generic_data(url, path)


def download_zip_data(url, path="data/"):
    """
    Example::

        from flash.core.data import download_zip_data

        # download titanic data
        download_zip_data("https://pl-flash.core.data.s3.amazonaws.com/imdb.zip", "data/")

    Args:
        url: must end with .zip
        path: path to download to

    Returns:
    """
    with urlopen(url) as resp:
        with ZipFile(BytesIO(resp.read())) as file:
            file.extractall(path)


def download_generic_data(url, path="data/"):
    """
    Downloads an arbitrary file.

    Example::

        from flash.core.data import download_csv_data

        # download titanic data
        download_csv_data("https://pl-flash.core.data.s3.amazonaws.com/titanic.csv", "titanic.csv")

    Args:
        url: must end with .csv
        path: path to download to (include the file name)

    Returns:
    """
    urlretrieve(url, path)


def _contains_any_tensor(value, dtype=torch.Tensor):
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
