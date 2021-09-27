from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tqdm import tqdm


def fn_outputs_to_keyed_map(serialize_fn_out_keys, fn_output) -> Dict[str, Any]:
    """convert outputs of a function to a dict of `{result_name: values}`

    accepts function outputs which are sequence, dict, or object.
    """
    if len(serialize_fn_out_keys) == 1:
        if not isinstance(fn_output, dict):
            fn_output = dict(zip(serialize_fn_out_keys, [fn_output]))
    elif not isinstance(fn_output, dict):
        fn_output = dict(zip(serialize_fn_out_keys, fn_output))
    return fn_output


def download_file(url: str, *, download_path: Optional[Path] = None) -> str:
    """Download to cwd with filename as last part of address, return filepath.

    Returns
    -------
    str
        Path to the downloaded file on disk
    download_path
        kwarg only which specifies the path to download the file to.
        By default, None.

    TODO
    ----
    *  cleanup on error
    *  allow specific file names
    """
    fname = f"{url.split('/')[-1]}"
    fpath = str(download_path.absolute()) if download_path is not None else f"./{fname}"

    response = requests.get(url, stream=True)
    nbytes = int(response.headers.get("content-length", 0))
    with tqdm.wrapattr(open(fpath, "wb"), "write", miniters=1, desc=fname, total=nbytes) as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    return fpath


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False
