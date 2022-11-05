.. _contributing_data:

********
The Data
********

The first step to contributing a task is to implement the classes we need to load some data.
Inside `data.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/template/classification/data.py>`_ you should implement:

#. some :class:`~flash.core.data.io.input.Input` classes *(optional)*
#. a :class:`~flash.core.data.io.input_transform.InputTransform`
#. a :class:`~flash.core.data.data_module.DataModule`
#. a :class:`~flash.core.data.base_viz.BaseVisualization` *(optional)*
#. a :class:`~flash.core.data.io.output_transform.OutputTransform` *(optional)*

Input
^^^^^

The :class:`~flash.core.data.io.input.Input` class contains the logic for data loading from different sources such as folders, files, tensors, etc.
Every Flash :class:`~flash.core.data.data_module.DataModule` can be instantiated with :meth:`~flash.core.data.data_module.DataModule.from_datasets`.
For each additional way you want the user to be able to instantiate your :class:`~flash.core.data.data_module.DataModule`, you'll need to create a :class:`~flash.core.data.io.input.Input`.
Each :class:`~flash.core.data.io.input.Input` has 2 methods:

- :meth:`~flash.core.data.io.input.Input.load_data` takes some dataset metadata (e.g. a folder name) as input and produces a sequence or iterable of samples or sample metadata.
- :meth:`~flash.core.data.io.input.Input.load_sample` then takes as input a single element from the output of ``load_data`` and returns a sample.

By default these methods just return their input, so you don't need both a :meth:`~flash.core.data.io.input.Input.load_data` and a :meth:`~flash.core.data.io.input.Input.load_sample` to create a :class:`~flash.core.data.io.input.Input`.
Where possible, you should override one of our existing :class:`~flash.core.data.io.input.Input` classes.

Let's start by implementing a ``TemplateNumpyClassificationInput``, which overrides :class:`~flash.core.data.io.classification_input.ClassificationInputMixin`.
The main :class:`~flash.core.data.io.input.Input` method that we have to implement is :meth:`~flash.core.data.io.input.Input.load_data`.
:class:`~flash.core.data.io.classification_input.ClassificationInputMixin` provides utilities for handling targets within flash which need to be called from the :meth:`~flash.core.data.io.input.Input.load_data` and :meth:`~flash.core.data.io.input.Input.load_sample`.
In this ``Input``, we'll also set the ``num_features`` attribute so that we can access it later.

Here's the code for our ``TemplateNumpyClassificationInput.load_data`` method:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateNumpyClassificationInput.load_data

and here's the code for the ``TemplateNumpyClassificationInput.load_sample`` method:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateNumpyClassificationInput.load_sample

.. note:: Later, when we add :ref:`our DataModule implementation <contributing_data_module>`, we'll make ``num_features`` available to the user.

For our template :class:`~flash.core.data.model.Task`, it would be cool if the user could provide a scikit-learn ``Bunch`` as the data source.
To achieve this, we'll add a ``TemplateSKLearnClassificationInput`` whose ``load_data`` expects a ``Bunch`` as input.
We override our ``TemplateNumpyClassificationInput`` so that we can call ``super`` with the data and targets extracted from the ``Bunch``.
We perform two additional steps here to improve the user experience:

1. We set the ``num_classes`` attribute on the ``dataset``. If ``num_classes`` is set, it is automatically made available as a property of the :class:`~flash.core.data.data_module.DataModule`.
2. We create and set a :class:`~flash.core.data.io.input.ClassificationState`. The labels provided here will be shared with the :class:`~flash.core.classification.Labels` output, so the user doesn't need to provide them.

Here's the code for the ``TemplateSKLearnClassificationInput.load_data`` method:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassificationInput.load_data

We can customize the behaviour of our :meth:`~flash.core.data.io.input.Input.load_data` for different stages, by prepending `train`, `val`, `test`, or `predict`.
For our ``TemplateSKLearnClassificationInput``, we don't want to provide any targets to the model when predicting.
We can implement ``predict_load_data`` like this:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassificationInput.predict_load_data

InputTransform
^^^^^^^^^^^^^^

The :class:`~flash.core.data.io.input_transform.InputTransform` object contains all the data transforms.
Internally we inject the :class:`~flash.core.data.io.input_transform.InputTransform` transforms at several points along the pipeline.

Defining the standard transforms (typically at least a ``per_sample_transform`` should be defined) for your :class:`~flash.core.data.io.input_transform.InputTransform` involves simply overriding the required hook to return a callable transform.

For our ``TemplateInputTransform``, we'll just configure a ``per_sample_transform``.
Let's first define a to_tensor transform as a ``staticmethod``:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateInputTransform.to_tensor

Now in our ``per_sample_transform`` hook, we return the transform:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateInputTransform.per_sample_transform

.. _contributing_data_module:

DataModule
^^^^^^^^^^

The :class:`~flash.core.data.data_module.DataModule` is responsible for creating the :class:`~torch.utils.data.DataLoader` and injecting the transforms for each stage.
When the user calls a ``from_*`` method (such as :meth:`~flash.core.data.data_module.DataModule.from_numpy`), the following steps take place:

#. The :meth:`~flash.core.data.data_module.DataModule.from_` method is called with the name of the :class:`~flash.core.data.io.input.Input` to use and the inputs to provide to :meth:`~flash.core.data.io.input.Input.load_data` for each stage.
#. The :class:`~flash.core.data.io.input_transform.InputTransform` is created from ``cls.input_transform_cls`` (if it wasn't provided by the user) with any provided transforms.
#. The :class:`~flash.core.data.io.input.Input` of the provided name is retrieved from the :class:`~flash.core.data.io.input_transform.InputTransform`.
#. A :class:`~flash.core.data.auto_dataset.BaseAutoDataset` is created from the :class:`~flash.core.data.io.input.Input` for each stage.
#. The :class:`~flash.core.data.data_module.DataModule` is instantiated with the data sets.

|

To create our ``TemplateData`` :class:`~flash.core.data.data_module.DataModule`, we first need to attach our input transform class like this:

.. code-block:: python

    input_transform_cls = TemplateInputTransform

Since we provided a :attr:`~flash.core.data.io.input.InputFormat.NUMPY` :class:`~flash.core.data.io.input.Input` in the ``TemplateInputTransform``, :meth:`~flash.core.data.data_module.DataModule.from_numpy` will now work with our ``TemplateData``.

If you've defined a fully custom :class:`~flash.core.data.io.input.Input` (like our ``TemplateSKLearnClassificationInput``), then you will need to write a ``from_*`` method for each.
Here's the ``from_sklearn`` method for our ``TemplateData``:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateData.from_sklearn

The final step is to implement the ``num_features`` property for our ``TemplateData``.
This is just a convenience for the user that finds the ``num_features`` attribute on any of the data sets and returns it.
Here's the code:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateData.num_features

BaseVisualization
^^^^^^^^^^^^^^^^^

An optional step is to implement a :class:`~flash.core.data.base_viz.BaseVisualization`.
The :class:`~flash.core.data.base_viz.BaseVisualization` lets you control how data at various points in the pipeline can be visualized.
This is extremely useful for debugging purposes, allowing users to view their data and understand the impact of their transforms.

.. note::
    Don't worry about implementing it right away, you can always come back and add it later!

Here's the code for our ``TemplateVisualization`` which just prints the data:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :pyobject: TemplateVisualization

We can configure our custom visualization in the ``TemplateData`` using :meth:`~flash.core.data.data_module.DataModule.configure_data_fetcher` like this:

.. literalinclude:: ../../../flash/template/classification/data.py
    :language: python
    :dedent: 4
    :pyobject: TemplateData.configure_data_fetcher

OutputTransform
^^^^^^^^^^^^^^^

:class:`~flash.core.data.io.output_transform.OutputTransform` contains any transforms that need to be applied *after* the model.
You may want to use it for: converting tokens back into text, applying an inverse normalization to an output image, resizing a generated image back to the size of the input, etc.
As an example, here's the :class:`~image.segmentation.model.SemanticSegmentationOutputTransform` which decodes tokenized model outputs:

.. literalinclude:: ../../../flash/image/segmentation/model.py
    :language: python
    :pyobject: SemanticSegmentationOutputTransform

In your :class:`~flash.core.data.io.input.Input` or :class:`~flash.core.data.io.input_transform.InputTransform`, you can add metadata to the batch using the :attr:`~flash.core.data.io.input.DataKeys.METADATA` key.
Your :class:`~flash.core.data.io.output_transform.OutputTransform` can then use this metadata in its transforms.
You should use this approach if your postprocessing depends on the state of the input before the :class:`~flash.core.data.io.input_transform.InputTransform` transforms.
For example, if you want to resize the predictions to the original size of the inputs you should add the original image size in the :attr:`~flash.core.data.io.input.DataKeys.METADATA`.
Here's an example from the :class:`~flash.image.data.ImageInput`:

.. literalinclude:: ../../../flash/image/data.py
    :language: python
    :dedent: 4
    :pyobject: ImageInput.load_sample

The :attr:`~flash.core.data.io.input.DataKeys.METADATA` can now be referenced in your :class:`~flash.core.data.io.output_transform.OutputTransform`.
For example, here's the code for the ``per_sample_transform`` method of the :class:`~flash.image.segmentation.model.SemanticSegmentationOutputTransform`:

.. literalinclude:: ../../../flash/image/segmentation/model.py
    :language: python
    :dedent: 4
    :pyobject: SemanticSegmentationOutputTransform.per_sample_transform

------

Now that you've got some data, it's time to :ref:`add some backbones for your task! <contributing_backbones>`
