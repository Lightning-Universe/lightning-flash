########
Callback
########

.. _callback:

**************
Flash Callback
**************

:class:`~flash.data.callback.FlashCallback` are extensions of the PyTorch Lightning `Callback <https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html>`__.

A callback is a self-contained program that can be reused across projects.

Flash and Lightning have a callback system to execute callbacks when needed.

Callbacks should capture NON-ESSENTIAL logic that is NOT required for your lightning module to run.

*******************
Available Callbacks
*******************


BaseDataFetcher
_______________

.. autoclass:: flash.data.callback.BaseDataFetcher
   :members: enable

BaseViz
_______

.. autoclass:: flash.data.base_viz.BaseViz
   :members:


*************
API reference
*************


FlashCallback
_____________

.. autoclass:: flash.data.callback.FlashCallback
    :members:
