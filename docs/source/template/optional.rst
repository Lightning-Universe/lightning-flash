.. _contributing_optional:

***************
Optional Extras
***************

Organize your transforms in transforms.py
=========================================

If you have a lot of default transforms, it can be useful to put them all in a ``transforms.py`` file, to be referenced in your :class:`~flash.core.data.process.Preprocess`.
Here's an example from `image/classification/transforms.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/classification/transforms.py>`_ which creates some default transforms given the desired image size:

.. literalinclude:: ../../../flash/image/classification/transforms.py
    :language: python
    :pyobject: default_transforms

Here's how we create our transforms in the :class:`~flash.image.classification.data.ImageClassificationPreprocess`:

.. literalinclude:: ../../../flash/image/classification/data.py
    :language: python
    :pyobject: ImageClassificationPreprocess.default_transforms

Add output serializers to your Task
======================================

Sometimes you want to give the user some control over their prediction format.
:class:`~flash.core.data.process.Postprocess` can do the heavy lifting (anything you always want to apply to the predictions), but one or more custom :class:`~flash.core.data.process.Serializer` implementations can be used to convert the predictions to a desired output format.
You should add your :class:`~flash.core.data.process.Serializer` implementations in a ``serialization.py`` file and set a good default in your :class:`~flash.core.model.Task`.
Some good examples are in `flash/core/classification.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/core/classification.py>`_.
Here's the :class:`~flash.core.classification.Classes` :class:`~flash.core.data.process.Serializer`:

.. literalinclude:: ../../../flash/core/classification.py
    :language: python
    :pyobject: Classes

Alternatively, here's the :class:`~flash.core.classification.Logits` :class:`~flash.core.data.process.Serializer`:

.. literalinclude:: ../../../flash/core/classification.py
    :language: python
    :pyobject: Logits

Take a look at :ref:`predictions` to learn more.

------

Once you've added any optional extras, it's time to :ref:`create some examples showing your task in action! <contributing_examples>`
