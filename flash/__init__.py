"""Root package info."""
import os

__version__ = "0.0.3"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"  # TODO
__license__ = "TBD"  # TODO
__copyright__ = f"Copyright (c) 2020-2021, f{__author__}."
__homepage__ = "https://github.com/PyTorchLightning/pytorch-lightning-flash"
__docs__ = "Flash is a framework for fast prototyping, finetuning, and solving most standard deep learning challenges"
__long_doc__ = """
Flash is a task-based deep learning framework for flexible deep learning built on PyTorch Lightning.
Tasks can be anything from text classification to object segmentation.
Although PyTorch Lightning provides ultimate flexibility, for common tasks it does not remove 100% of the boilerplate.
Flash is built for applied researchers, beginners, data scientists, Kagglers or anyone starting out with Deep Learning.
But unlike other entry-level frameworks (keras, etc...), Flash users can switch to Lightning trivially when they need
the added flexibility.
"""

PACKAGE_ROOT = os.path.dirname(__file__)

try:
    # This variable is injected in the __builtins__ by the build process.
    # It used to enable importing subpackages when the binaries are not built.
    __LIGHTNING_FLASH_SETUP__
except NameError:
    __LIGHTNING_FLASH_SETUP__ = False

if __LIGHTNING_FLASH_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:

    from flash import tabular, text, vision
    from flash.core import data, utils
    from flash.core.data import DataModule
    from flash.core.model import ClassificationTask, Task

    __all__ = ["Task", "ClassificationTask", "DataModule", "vision", "text", "tabular", "data", "utils"]
