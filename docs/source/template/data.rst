.. _contributing_data:

********
The Data
********

The first step to contributing a task is to implement the classes we need to load some data.
Inside ``data.py`` you should implement:

#. one or more :class:`~flash.core.data.data_source.DataSource` classes
#. a :class:`~flash.core.data.process.Preprocess`
#. a :class:`~flash.core.data.data_module.DataModule`
#. a :class:`~flash.core.data.callbacks.BaseVisualization` *(optional)*
#. a :class:`~flash.core.data.process.Postprocess` *(optional)*

DataSource
^^^^^^^^^^

The :class:`~flash.core.data.data_source.DataSource` implementations describe how data from particular sources (like folders, files, tensors, etc.) should be loaded.
At a minimum you will require one :class:`~flash.core.data.data_source.DataSource` implementation, but if you want to support a few different ways of loading data for your task, the more the merrier!
Each :class:`~flash.core.data.data_source.DataSource` has a ``load_data`` method and a ``load_sample`` method.
The ``load_data`` method accepts some dataset metadata (e.g. a folder name) and produces a sequence or iterable of samples or sample metadata.
The ``load_sample`` method then takes as input a single element from the output of ``load_data`` and returns a sample.
By default, these methods just return their input as you will not always need both to create a functioning :class:`~flash.core.data.data_source.DataSource`.

I'ts best practice to just override one of our existing :class:`~flash.core.data.data_source.DataSource` classes where possible.
Take a look at our ``TemplateNumpyDataSource`` which does this to get started:

.. autoclass:: flash.template.data.TemplateNumpyDataSource
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../../flash/template/data.py
    :language: python
    :pyobject: TemplateNumpyDataSource

.. raw:: html

    </details>

|

Sometimes you need to something a bit more custom, have a look at our ``TemplateSKLearnDataSource`` for an example:

.. autoclass:: flash.template.data.TemplateSKLearnDataSource
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../../flash/template/data.py
    :language: python
    :pyobject: TemplateSKLearnDataSource

.. raw:: html

    </details>

DataSource vs Dataset
~~~~~~~~~~~~~~~~~~~~~

A :class:`~flash.core.data.data_source.DataSource` is not the same as a :class:`torch.utils.data.Dataset`.
A :class:`torch.utils.data.Dataset` knows about the data, whereas a :class:`~flash.core.data.data_source.DataSource` only know about how to load the data.
So it's possible for a single :class:`~flash.core.data.data_source.DataSource` to create more than one :class:`~torch.utils.data.Dataset`.
It's also fine for the output of the ``load_data`` method to just be a :class:`torch.utils.data.Dataset` instance.
You don't need to re-write custom datasets, just instantiate them in ``load_data`` similarly to what we did with the ``TemplateSKLearnDataSource``.
For example, the ``load_data`` of the ``VideoClassificationPathsDataSource`` just creates an :class:`~pytorchvideo.data.EncodedVideoDataset` from the given folder.
Here's how it looks (from ``video/classification.data.py``):

.. literalinclude:: ../../../flash/video/classification/data.py
    :language: python
    :pyobject: VideoClassificationPathsDataSource.load_data

Preprocess
^^^^^^^^^^

The :class:`~flash.core.data.process.Preprocess` is how all transforms are defined in Flash.
Internally we inject the :class:`~flash.core.data.process.Preprocess` transforms into the right places so that we can address the batch at several points along the pipeline.
Defining the standard transforms (typically at least a ``to_tensor_transform`` should be defined) for your :class:`~flash.core.data.process.Preprocess` is as simple as implementing the ``default_transforms`` method.
The :class:`~flash.core.data.process.Preprocess` also knows about the available :class:`~flash.core.data.data_source.DataSource` classes that it can work with, which should be configured in the ``__init__``.

Take a look at our ``TemplatePreprocess`` to get started:

.. autoclass:: flash.template.data.TemplatePreprocess
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../../flash/template/data.py
    :language: python
    :pyobject: TemplatePreprocess

.. raw:: html

    </details>

DataModule
^^^^^^^^^^

The :class:`~flash.core.data.data_module.DataModule` is where the hard work of our :class:`~flash.core.data.data_source.DataSource` and :class:`~flash.core.data.process.Preprocess` implementations pays off.
If your :class:`~flash.core.data.data_source.DataSource` implementation(s) conform to our :class:`~flash.core.data.data_source.DefaultDataSources` (e.g. ``DefaultDataSources.FOLDERS``) then your :class:`~flash.core.data.data_module.DataModule` implementation simply needs a ``preprocess_cls`` attribute.
You now have a :class:`~flash.core.data.data_module.DataModule` that can be instantiated with ``from_*`` for whichever data sources you have configured (e.g. ``MyDataModule.from_folders``).
It also includes all of your default transforms!

If you've defined a fully custom :class:`~flash.core.data.data_source.DataSource` (like our ``TemplateSKLearnDataSource``), then you will need a ``from_*`` method for each (we'll define ``from_sklearn`` for our example).
The ``from_*`` methods take whatever arguments you want them to and call :meth:`~flash.core.data.data_module.DataModule.from_data_source` with the name given to your custom data source in the ``Preprocess.__init__``.


Take a look at our ``TemplateData`` to get started:

.. autoclass:: flash.template.data.TemplateData
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../../flash/template/data.py
    :language: python
    :pyobject: TemplateData

.. raw:: html

    </details>

BaseVisualization
^^^^^^^^^^^^^^^^^

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

.. literalinclude:: ../../../flash/template/data.py
    :language: python
    :pyobject: TemplateVisualization

.. raw:: html

    </details>

Postprocess
^^^^^^^^^^^

Sometimes you have some transforms that need to be applied _after_ your model.
For this you can optionally implement a :class:`~flash.core.data.process.Postprocess`.
The :class:`~flash.core.data.process.Postprocess` is applied to the model outputs during inference.
You may want to use it for: converting tokens back into text, applying an inverse normalization to an output image, resizing a generated image back to the size of the input, etc.

------

Now that you've got some data, it's time to :ref:`implement your task! <contributing_task>`
