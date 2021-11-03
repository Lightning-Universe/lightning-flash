.. customcarditem::
   :header: Semantic Segmentation
   :card_description: Learn to segment images with Flash and build a model which segments images from the CARLA driving simulator.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/semantic_segmentation.svg
   :tags: Image,Segmentation

.. _semantic_segmentation:

#####################
Semantic Segmentation
#####################

********
The Task
********
Semantic Segmentation, or image segmentation, is the task of performing classification at a pixel-level, meaning each pixel will associated to a given class.
See more: https://paperswithcode.com/task/semantic-segmentation

------

*******
Example
*******

Let's look at an example using a data set generated with the `CARLA <http://carla.org/>`_ driving simulator.
The data was generated as part of the `Kaggle Lyft Udacity Challenge <https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge>`_.
The data contains one folder of images and another folder with the corresponding segmentation masks.
Here's the structure:

.. code-block::

    data
    ├── CameraRGB
    │   ├── F61-1.png
    │   ├── F61-2.png
    │       ...
    └── CameraSeg
        ├── F61-1.png
        ├── F61-2.png
            ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.segmentation.data.SemanticSegmentationData`.
We select a pre-trained ``mobilenet_v3_large`` backbone with an ``fpn`` head to use for our :class:`~flash.image.segmentation.model.SemanticSegmentation` task and fine-tune on the CARLA data.
We then use the trained :class:`~flash.image.segmentation.model.SemanticSegmentation` for inference. You can check the available pretrained weights for the backbones like this `SemanticSegmentation.available_pretrained_weights("resnet18")`.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/semantic_segmentation.py
    :language: python
    :lines: 14-


------

**********
Flash Zero
**********

The semantic segmentation task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash semantic_segmentation

To view configuration options and options for running the semantic segmentation task with your own data, use:

.. code-block:: bash

    flash semantic_segmentation --help

------

************
Loading Data
************

.. autoinputs:: flash.image.segmentation.data SemanticSegmentationData

    {% extends "base.rst" %}
    {% block from_folders %}
    Construct the {{ data_module }} from folders.

    {% if data_sources['folders'].extensions is defined %}
    The supported file extensions are: {{ data_sources['folders'].extensions|join(', ') }}.
    {% set extension = data_sources['folders'].extensions[0] %}
    {% else %}
    {% set extension = '' %}
    {% endif %}

    For train, test, and val data, we expect a folder containing inputs and another folder containing the masks.
    Here's the required structure:

    .. code-block::

        train_folder
        ├── inputs
        │   ├── file1{{ extension }}
        │   ├── file2{{ extension }}
        │   ...
        └── masks
            ├── file1{{ extension }}
            ├── file2{{ extension }}
            ...

    For prediction, the folder is expected to contain the files for inference, like this:

    .. code-block::

        predict_folder
        ├── file1{{ extension }}
        ├── file2{{ extension }}
        ...

    Example::

        data_module = {{ data_module_raw }}.from_folders(
            train_folder = "./train_folder/inputs",
            train_target_folder = "./train_folder/masks",
            predict_folder = "./predict_folder",
            ...
        )
    {% endblock %}
    {% block from_files %}
    Construct the {{ data_module }} from lists of input images and corresponding list of target images.

    {% if data_sources['files'].extensions is defined %}
    The supported file extensions are: {{ data_sources['files'].extensions|join(', ') }}.
    {% set extension = data_sources['files'].extensions[0] %}
    {% else %}
    {% set extension = '' %}
    {% endif %}

    Example::

        train_files = ["file1{{ extension }}", "file2{{ extension }}", "file3{{ extension }}", ...]
        train_targets = ["mask1{{ extension }}", "mask2{{ extension }}", "mask3{{ extension }}", ...]

        datamodule = {{ data_module_raw }}.from_files(
            train_files = train_files,
            train_targets = train_targets,
            ...
        )
    {% endblock %}
    {% block from_datasets %}
    {{ super() }}

    .. note::

        The ``__getitem__`` of your datasets should return a dictionary with ``"input"`` and ``"target"`` keys which map to the input and target images as tensors.
    {% endblock %}

------

*******
Serving
*******

The :class:`~flash.image.segmentation.model.SemanticSegmentation` task is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/semantic_segmentation/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/semantic_segmentation/client.py
    :language: python
    :lines: 14-
