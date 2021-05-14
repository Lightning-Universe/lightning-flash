Task Template
=============

This template is designed to guide you through implementing your own task in flash.
You should copy the files in ``flash/template`` and adapt them to your custom task.

.. contents:: Contents:
    :local:
    :depth: 3

Required Files
--------------

``data.py``
~~~~~~~~~~~

Inside ``data.py`` you should implement:

#. one or more :class:`~flash.data.data_source.DataSource` classes
#. a :class:`~flash.data.process.Preprocess`
#. a :class:`~flash.data.data_module.DataModule`
#. a :class:`~flash.data.callbacks.BaseVisualization` *(optional)*
#. a :class:`~flash.data.process.Postprocess` *(optional)*

``DataSource``
^^^^^^^^^^^^^^

The :class:`~flash.data.data_source.DataSource` implementations describe how data from particular sources (like folders, files, tensors, etc.) should be loaded.
At a minimum you will require one :class:`~flash.data.data_source.DataSource` implementation, but if you want to support a few different ways of loading data for your task, the more the merrier!

Take a look at our ``TemplateNumpyDataSource`` to get started:

.. autoclass:: flash.template.data.TemplateNumpyDataSource
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../flash/template/data.py
    :language: python
    :pyobject: TemplateNumpyDataSource

.. raw:: html

    </details>

|

And have a look at our ``TemplateSKLearnDataSource`` for another example:

.. autoclass:: flash.template.data.TemplateSKLearnDataSource
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../flash/template/data.py
    :language: python
    :pyobject: TemplateSKLearnDataSource

.. raw:: html

    </details>

``Preprocess``
^^^^^^^^^^^^^^

The :class:`~flash.data.process.Preprocess` is how all transforms are defined in Flash.
Internally we inject the :class:`~flash.data.process.Preprocess` transforms into the right places so that we can address the batch at several points along the pipeline.
Defining the standard transforms (typically at least a ``to_tensor_transform`` should be defined) for your :class:`~flash.data.process.Preprocess` is as simple as implementing the ``default_transforms`` method.
The :class:`~flash.data.process.Preprocess` also knows about the available :class:`~flash.data.data_source.DataSource` classes that it can work with, which should be configured in the ``__init__``.

Take a look at our ``TemplatePreprocess`` to get started:

.. autoclass:: flash.template.data.TemplatePreprocess
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../flash/template/data.py
    :language: python
    :pyobject: TemplatePreprocess

.. raw:: html

    </details>

``DataModule``
^^^^^^^^^^^^^^

The :class:`~flash.data.data_module.DataModule` is where the hard work of our :class:`~flash.data.data_source.DataSource` and :class:`~flash.data.process.Preprocess` implementations pays off.
If your :class:`~flash.data.data_source.DataSource` implementation(s) conform to our :class:`~flash.data.data_source.DefaultDataSources` (e.g. ``DefaultDataSources.FOLDERS``) then your :class:`~flash.data.data_module.DataModule` implementation simply needs a ``preprocess_cls`` attribute.
You now have a :class:`~flash.data.data_module.DataModule` that can be instantiated with ``from_*`` for whichever data sources you have configured (e.g. ``MyDataModule.from_folders``).
It also includes all of your default transforms!

If you've defined a fully custom :class:`~flash.data.data_source.DataSource` (like our ``TemplateSKLearnDataSource``), then you will need a ``from_*`` method for each (we'll define ``from_sklearn`` for our example).
The ``from_*`` methods take whatever arguments you want them too and call ``super().from_data_source`` with the name given to your custom data source in the ``Preprocess.__init__``.


Take a look at our ``TemplateData`` to get started:

.. autoclass:: flash.template.data.TemplateData
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../flash/template/data.py
    :language: python
    :pyobject: TemplateData

.. raw:: html

    </details>

``BaseVisualization``
^^^^^^^^^^^^^^^^^^^^^

An optional step is to implement a ``BaseVisualization``. The ``BaseVisualization`` lets you control how data at various points in the pipeline can be visualized.
This is extremely useful for debugging purposes, allowing users to view their data and understand the impact of their transforms.

Take a look at our ``TemplateVisualization`` to get started:

.. note::
    Don't worry about implementing it right away, you can always come back and add it later!

.. autoclass:: flash.template.data.TemplateVisualization
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../flash/template/data.py
    :language: python
    :pyobject: TemplateVisualization

.. raw:: html

    </details>

``Postprocess``
^^^^^^^^^^^^^^^

Sometimes you have some transforms that need to be applied _after_ your model.
For this you can optionally implement a :class:`~flash.data.process.Postprocess`.
The :class:`~flash.data.process.Postprocess` is applied to the model outputs during inference.
You may want to use it for: converting tokens back into text, applying an inverse normalization to an output image, resizing a generated image back to the size of the input, etc.

`model.py`
~~~~~~~~~~

Inside `model.py` you just need to implement your `Task`.
