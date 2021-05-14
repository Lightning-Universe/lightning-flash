########
Callback
########

.. _callback:

**************
Flash Callback
**************

:class:`~flash.core.data.callback.FlashCallback` is an extension of :class:`pytorch_lightning.callbacks.Callback`.

A callback is a self-contained program that can be reused across projects.

Flash and Lightning have a callback system to execute callbacks when needed.

Callbacks should capture any NON-ESSENTIAL logic that is NOT required for your lightning module to run.

Same as PyTorch Lightning, Callbacks can be provided directly to the Trainer.

Example::

   trainer = Trainer(callbacks=[MyCustomCallback()])


*******************
Available Callbacks
*******************


BaseDataFetcher
_______________

.. autoclass:: flash.core.data.callback.BaseDataFetcher
   :members: enable

BaseVisualization
_________________

.. autoclass:: flash.core.data.base_viz.BaseVisualization
   :members:


*************
API reference
*************


FlashCallback
_____________

.. autoclass:: flash.core.data.callback.FlashCallback
    :members:
