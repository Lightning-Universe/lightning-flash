#!/usr/bin/env python
import os
import subprocess
# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

# Used to install Lightning Master
"""
try:
    import pytorch_lightning
except (ImportError, AssertionError):
    subprocess.Popen(["pip", "install", "git+https://github.com/PyTorchLightning/pytorch-lightning.git"])
"""

try:
    import flash.__about__ as about  # noqa: E402
    import flash.setup_tools  # noqa: E402
except ImportError:
    # see https://stackoverflow.com/a/129374
    sys.path.append("flash")
    import __about__ as about
    import setup_tools

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

_PATH_ROOT = os.path.dirname(__file__)

long_description = setup_tools._load_readme_description(_PATH_ROOT, homepage=about.__homepage__, ver=about.__version__)

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
    packages=find_packages(exclude=["tests", "docs"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=setup_tools._load_requirements(_PATH_ROOT, file_name='requirements.txt'),
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
