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
import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_py_module(fname, pkg="flash"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname),
        os.path.join(_PATH_ROOT, pkg, fname),
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")
setup_tools = _load_py_module("setup_tools.py")

long_description = setup_tools._load_readme_description(
    _PATH_ROOT,
    homepage=about.__homepage__,
    ver=about.__version__,
)

extras = {
    "docs": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="docs.txt"),
    "notebooks": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="notebooks.txt"),
    "test": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="test.txt"),
    "text": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_text.txt"),
    "tabular": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_tabular.txt"),
    "image": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_image.txt"),
    "image_extras": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_image_extras.txt"),
    "video": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_video.txt"),
    "pointcloud": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_pointcloud.txt"),
    "video_extras": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_video_extras.txt"),
    "serve": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="serve.txt"),
    "audio": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_audio.txt"),
    "graph": setup_tools._load_requirements(path_dir=_PATH_REQUIRE, file_name="datatype_graph.txt"),
}

extras["vision"] = list(set(extras["image"] + extras["video"]))
extras["all"] = list(
    set(extras["vision"] + extras["tabular"] + extras["text"])
)  # + extras["pointcloud"] dependencies conflicts
extras["dev"] = list(set(extras["all"] + extras["test"] + extras["docs"]))

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
    download_url="https://github.com/PyTorchLightning/lightning-flash",
    license=about.__license__,
    packages=find_packages(exclude=["tests", "tests.*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    extras_require=extras,
    entry_points={
        "console_scripts": ["flash=flash.__main__:main"],
    },
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.6",
    install_requires=setup_tools._load_requirements(_PATH_ROOT, file_name="requirements.txt"),
    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/lightning-flash/issues",
        "Documentation": "https://lightning-flash.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/lightning-flash",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
