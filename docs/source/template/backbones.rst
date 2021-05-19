.. _contributing_backbones:

*************
The Backbones
*************

Now that you've got a way of loading data, you should implement some backbones to use with your :class:`~flash.core.model.Task`.
Create a :any:`FlashRegistry <registry>` to use with your :class:`~flash.core.model.Task` in `backbones.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/template/classification/backbones.py>`_.

The registry allows you to register backbones for your task that can be selected by the user.
The backbones can come from anywhere as long as you can register a function that loads the backbone.
Furthermore, the user can add their own models to the existing backbones, without having to write their own :class:`~flash.core.model.Task`!

You can create a registry like this:

.. code-block:: python

    TEMPLATE_BACKBONES = FlashRegistry("backbones")

Let's add a simple MLP backbone to our registry.
We'll create the backbone and return it along with the output size (so that we can create the model head in our :class:`~flash.core.model.Task`).
Here's the code:

.. literalinclude:: ../../../flash/template/classification/backbones.py
    :language: python
    :pyobject: load_mlp_128

Here's another example with a slightly more complex model:

.. literalinclude:: ../../../flash/template/classification/backbones.py
    :language: python
    :pyobject: load_mlp_128_256

Here's a more advanced example, which adds ``SimCLR`` to the ``IMAGE_CLASSIFIER_BACKBONES``, from `flash/image/backbones.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/backbones.py>`_:

.. literalinclude:: ../../../flash/image/backbones.py
    :language: python
    :pyobject: load_simclr_imagenet

------

Once you've got some data and some backbones, :ref:`implement your task! <contributing_task>`
