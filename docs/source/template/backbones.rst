.. _contributing_backbones:

*************
The Backbones
*************

Now that you've got a way of loading data, you should implement some backbones to use with your :class:`~flash.core.model.Task`.
Create a :class:`~flash.core.registry.FlashRegistry` to use with your :class:`~flash.core.model.Task` in `backbones.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/template/classification/backbones.py>`_.

The registry allows you to register backbones for your task that can be selected by the user.
The backbones can come from anywhere as long as you can register a function that loads the backbone.
Furthermore, the user can add their own models to the existing backbones, without having to write their own :class:`~flash.core.model.Task`!

You can create a registry like this:

.. code-block:: python

    TEMPLATE_BACKBONES = FlashRegistry("backbones")

Let's add a simple MLP backbone to our registry.
We need a function that creates the backbone and returns it along with the output size (so that we can create the model head in our :class:`~flash.core.model.Task`).
You can use any name for the function, although we use ``load_{model name}`` by convention.
You also need to provide ``name`` and ``namespace`` of the backbone.
The standard for *namespace* is ``data_type/task_type``, so for an image classification task the namespace will be ``image/classification``.
Here's the code:

.. literalinclude:: ../../../src/flash/template/classification/backbones.py
    :language: python
    :pyobject: load_mlp_128

Here's another example with a slightly more complex model:

.. literalinclude:: ../../../src/flash/template/classification/backbones.py
    :language: python
    :pyobject: load_mlp_128_256

Here's a another example, which adds ``DINO`` pretrained model from PyTorch Hub to the ``IMAGE_CLASSIFIER_BACKBONES``, from `flash/image/classification/backbones/transformers.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/classification/backbones/transformers.py>`_:

.. literalinclude:: ../../../src/flash/image/classification/backbones/transformers.py
    :language: python
    :pyobject: dino_vitb16

------

Once you've got some data and some backbones, :ref:`implement your task! <contributing_task>`
