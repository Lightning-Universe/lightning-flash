"""Root package info."""
import os

__version__ = "0.0.3"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = "TBD"
__copyright__ = "Copyright (c) 2020-2020, %s." % __author__
__homepage__ = "https://github.com/PyTorchLightning/pytorch-lightning-flash"
__docs__ = "PyTorch Lightning flash is a simple training framework for fast research iterations"
__long_doc__ = """
flash is a simplified version of PyTorch Lightning. It automates the model training with good
defaults for different tasks.
Use `flash` when training a model with good defaults and fast iteration, since you don't
have to take care of all the update stuff - we do it for you!
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
