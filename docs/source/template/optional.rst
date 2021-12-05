.. _contributing_optional:

***************
Optional Extras
***************

Organize your transforms in transforms.py
=========================================

If you have a lot of default transforms, it can be useful to put them all in a ``transforms.py`` file, to be referenced in your :class:`~flash.core.data.io.input_transform.InputTransform`.
Here's an example from `image/classification/transforms.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/classification/transforms.py>`_ which creates some default transforms given the desired image size:

.. literalinclude:: ../../../flash/image/classification/transforms.py
    :language: python
    :pyobject: default_transforms

Here's how we create our transforms in the :class:`~flash.image.classification.data.ImageClassificationInputTransform`:

.. literalinclude:: ../../../flash/image/classification/data.py
    :language: python
    :pyobject: ImageClassificationInputTransform.default_transforms

Add outputs to your Task
========================

We recommend that you do most of the heavy lifting in the :class:`~flash.core.data.io.output_transform.OutputTransform`.
Specifically, it should include any formatting and transforms that should always be applied to the predictions.
If you want to support different use cases that require different prediction formats, you should add some :class:`~flash.core.data.io.output.Output` implementations in an ``output.py`` file.

Some good examples are in `flash/core/classification.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/core/classification.py>`_.
Here's the :class:`~flash.core.classification.ClassesOutput` :class:`~flash.core.data.io.output.Output`:

.. literalinclude:: ../../../flash/core/classification.py
    :language: python
    :pyobject: ClassesOutput

Alternatively, here's the :class:`~flash.core.classification.LogitsOutput` :class:`~flash.core.data.io.output.Output`:

.. literalinclude:: ../../../flash/core/classification.py
    :language: python
    :pyobject: LogitsOutput

Take a look at :ref:`predictions` to learn more.

------

Once you've added any optional extras, it's time to :ref:`create some examples showing your task in action! <contributing_examples>`
