#!/usr/bin/env python
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
import glob
import os
import re
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_readme_description(path_dir: str, homepage: str, ver: str) -> str:
    """Load readme as decribtion.

    >>> _load_readme_description(_PATH_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    text = open(path_readme, encoding="utf-8").read()

    # https://github.com/Lightning-AI/lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", ver)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={ver}")
    text = text.replace("pytorch-lightning.readthedocs.io/en/stable/", f"pytorch-lightning.readthedocs.io/en/{ver}")
    # codecov badge
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{ver}/graph/badge.svg")
    # replace github badges for release ones
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={ver}")

    return text


def _augment_requirement(ln: str, comment_char: str = "#", unfreeze: bool = True) -> str:
    """Adjust the upper version contrains.

    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # anything", unfreeze=False)
    'arrow<=1.2.2,>=1.2.0'
    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # strict", unfreeze=False)
    'arrow<=1.2.2,>=1.2.0  # strict'
    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # my name", unfreeze=True)
    'arrow>=1.2.0'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # strict", unfreeze=True)
    'arrow>=1.2.0, <=1.2.2  # strict'
    >>> _augment_requirement("arrow", unfreeze=True)
    'arrow'
    """
    # filer all comments
    if comment_char in ln:
        comment = ln[ln.index(comment_char) :]
        ln = ln[: ln.index(comment_char)]
        is_strict = "strict" in comment
    else:
        is_strict = False
    req = ln.strip()
    # skip directly installed dependencies
    if not req or any(c in req for c in ["http:", "https:", "@"]):
        return ""

    # remove version restrictions unless they are strict
    if unfreeze and "<" in req and not is_strict:
        req = re.sub(r",? *<=? *[\d\.\*]+,? *", "", req).strip()

    # adding strict back to the comment
    if is_strict:
        req += "  # strict"

    return req


def _load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: bool = not _FREEZE_REQUIREMENTS) -> list:
    """Loading requirements from a file.

    >>> path_req = os.path.join(_PATH_ROOT, "requirements")
    >>> _load_requirements(path_req, "docs.txt")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx>=4.0', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [_augment_requirement(ln, unfreeze=unfreeze) for ln in lines]
    reqs = [str(req) for req in reqs if req and not req.startswith("-r")]
    # filter empty lines and containing @ which means redirect to some git/http
    reqs = [req for req in reqs if not any(c in req for c in ["@", "http://", "https://"])]
    return reqs


def _load_py_module(fname, pkg="flash"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")


def _expand_reqs(extras: dict, keys: list) -> list:
    return list(chain(*[extras[ex] for ex in keys]))


# find all extra requirements
def _get_extras(path_dir: str = _PATH_REQUIRE) -> dict:
    _load_req = partial(_load_requirements, path_dir=path_dir)
    found_req_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path_dir, "*.txt")))
    found_req_files = [p for p in found_req_files if not p.startswith("testing_")]
    # remove datatype prefix
    found_req_names = [os.path.splitext(req)[0].replace("datatype_", "") for req in found_req_files]
    # define basic and extra extras
    extras_req = {
        name: _load_req(file_name=fname) for name, fname in zip(found_req_names, found_req_files) if "_" not in name
    }
    extras_req.update(
        {
            name: extras_req[name.split("_")[0]] + _load_req(file_name=fname)
            for name, fname in zip(found_req_names, found_req_files)
            if "_" in name
        }
    )
    # some extra combinations
    extras_req["vision"] = _expand_reqs(extras_req, ["image", "video"])
    extras_req["core"] = _expand_reqs(extras_req, ["image", "tabular", "text"])
    extras_req["all"] = _expand_reqs(extras_req, ["vision", "tabular", "text", "audio"])
    extras_req["dev"] = _expand_reqs(extras_req, ["all", "test", "docs"])
    # filter the uniques
    extras_req = {n: list(set(req)) for n, req in extras_req.items()}
    return extras_req


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="lightning-flash",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Lightning-AI/lightning-flash",
    license=about.__license__,
    packages=find_packages(exclude=["tests", "tests.*"]),
    long_description=_load_readme_description(_PATH_ROOT, homepage=about.__homepage__, ver=about.__version__),
    long_description_content_type="text/markdown",
    include_package_data=True,
    entry_points={
        "console_scripts": ["flash=flash.__main__:main"],
    },
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.7",
    install_requires=_load_requirements(path_dir=_PATH_ROOT, file_name="requirements.txt"),
    extras_require=_get_extras(),
    project_urls={
        "Bug Tracker": "https://github.com/Lightning-AI/lightning-flash/issues",
        "Documentation": "https://lightning-flash.rtfd.io/en/latest/",
        "Source Code": "https://github.com/Lightning-AI/lightning-flash",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        # 'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
