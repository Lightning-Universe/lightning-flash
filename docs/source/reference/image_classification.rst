
.. _image_classification:

####################
Image Classification
####################

********
The Task
********

The task of identifying what is in an image is called image classification.
Typically, Image Classification is used to identify images containing a single object.
The task predicts which ‘class’ the image most likely belongs to with a degree of certainty.
A class is a label that describes what is in an image, such as ‘car’, ‘house’, ‘cat’ etc.

------

*******
Example
*******

Let's look at the task of predicting whether images contain Ants or Bees using the hymenoptera dataset.
The dataset contains ``train`` and ``validation`` folders, and then each folder contains a **bees** folder, with pictures of bees, and an **ants** folder with images of, you guessed it, ants.

.. code-block::

    hymenoptera_data
    ├── train
    │   ├── ants
    │   │   ├── 0013035.jpg
    │   │   ├── 1030023514_aad5c608f9.jpg
    │   │   ...
    │   └── bees
    │       ├── 1092977343_cb42b38d62.jpg
    │       ├── 1093831624_fb5fbe2308.jpg
    │       ...
    └── val
        ├── ants
        │   ├── 10308379_1b6c72e180.jpg
        │   ├── 1053149811_f62a3410d3.jpg
        │   ...
        └── bees
            ├── 1032546534_06907fe3b3.jpg
            ├── 10870992_eebeeb3a12.jpg
            ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.classification.data.ImageClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.image.classification.model.ImageClassifier` and fine-tune on the hymenoptera data.
We then use the trained :class:`~flash.image.classification.model.ImageClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/image_classification.py
    :language: python
    :lines: 14-


------

*******
Custom Transformations
*******

Flash automatically takes care of Image transformations for you. But sometimes it is not enough and you want to customize the transformations.
The base :class:`~flash.core.data.process.Preprocess` defines 7 hooks for different stages in the data loading pipeline.
To apply image augmentations you can directly import the ``default_transforms`` from ``flash.image.classification.transforms`` and then merge your custom image transformations with them.
In this example, we will load the default transforms and merge with custom `torchvision` transformations. We use the `post_tensor_transform` hook to apply the transformations
after the image is converted to a `torch.Tensor`.


.. testcode:: transformations
    from torchvision import transforms as T

    from flash.core.data.data_source import DefaultDataKeys
    from flash.core.data.transforms import ApplyToKeys, merge_transforms
    from flash.core.data.utils import download_data
    from flash.image import ImageClassificationData
    from flash.image.classification.transforms import default_transforms

    image_size = (256, 256)
    default_image_transforms = default_transforms(image_size)

    post_tensor_transform = ApplyToKeys(
        DefaultDataKeys.INPUT,
        T.Compose([T.RandomHorizontalFlip(),
                   T.ColorJitter(),
                   T.RandomAutocontrast(),
                   T.RandomPerspective()])
    )

    new_transforms = merge_transforms(default_image_transforms, {"post_tensor_transform": post_tensor_transform})

    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        train_transform=new_transforms
    )


.. testoutput:: finetune
    :hide:

    ...

------

**********
Flash Zero
**********

The image classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the hymenoptera example with:

.. code-block:: bash

    flash image-classification

To view configuration options and options for running the image classifier with your own data, use:

.. code-block:: bash

    flash image-classification --help

------

*******
Serving
*******

The :class:`~flash.image.classification.model.ImageClassifier` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/image_classification/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/image_classification/client.py
    :language: python
    :lines: 14-
