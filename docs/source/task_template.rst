Task Template
=============

This template is designed to guide you through implementing your own task in flash.
You should copy the files in ``flash/template`` and adapt them to your custom task.

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

.. raw:: html

    <details>
    <summary>Click to expand</summary>

.. autoclass:: flash.template.data.TemplateNumpyDataSource
    :members:

.. raw:: html

    </details>

And have a look at our ``TemplateSKLearnDataSource`` for another example:

.. raw:: html

    <details>
    <summary>Click to expand</summary>

.. autoclass:: flash.template.data.TemplateSKLearnDataSource
    :members:

.. raw:: html

    </details>

``Preprocess``
^^^^^^^^^^^^^^

The ``Preprocess`` is how all transforms are defined in Flash.
Internally we inject the ``Preprocess`` transforms into the right places so that we can address the batch at several points along the pipeline.
Defining the standard transforms (typically at least a ``to_tensor_transform`` should be defined) for your ``Preprocess`` is as simple as implementing the ``default_transforms`` method.
The ``Preprocess`` also knows about the available `DataSource` classes that it can work with, which should be configured in the ``__init__``.

Take a look at our ``TemplatePreprocess`` to get started:

.. raw:: html

    <details>
    <summary>Click to expand</summary>

.. autoclass:: flash.template.data.TemplatePreprocess
    :members:

.. raw:: html

    </details>
