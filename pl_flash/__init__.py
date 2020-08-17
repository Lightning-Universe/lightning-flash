"""Root package info."""

import os

__version__ = '0.1.0'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'TBD'
__copyright__ = 'Copyright (c) 2020-2020, %s.' % __author__
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning-flash'
__docs__ = "PyTorch Lightning flash is a community contribution for ML researchers."
__long_doc__ = """
What is it?
-----------
flash is a simplified version of PyTorch Lightning. It automates the model training with good defaults for different tasks.

Use `pl_flash` when training a model with good defaults and fast iteration, since you don't have to take care of all the update stuff - we do it for you!

"""

PACKAGE_ROOT = os.path.dirname(__file__)

try:
    # This variable is injected in the __builtins__ by the build process.
    # It used to enable importing subpackages when the binaries are not built.
    __LIGHTNING_BOLT_SETUP__
except NameError:
    __LIGHTNING_BOLT_SETUP__ = False

if __LIGHTNING_BOLT_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:

    from pl_flash.models.mnist_module import LitMNIST
    from pl_flash import models, metrics, loggers, callbacks, datamodules, transforms

    __all__ = [
        'LitMNIST',
        'models',
        'metrics',
        'loggers',
        'callbacks',
        'datamodules'
    ]