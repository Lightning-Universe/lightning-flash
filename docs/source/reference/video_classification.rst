
.. _video_classification:

####################
Video Classification
####################

********
The task
********

Typically, Video Classification is used to identify video clips containing a single object.
The task predicts which ‘class’ the video clip most likely belongs to with a degree of certainty.
A class is a label that describes what is in the video clip, such as ‘car’, ‘house’, ‘cat’ etc.
For example, we can train the video classifier task on video clips with human actions
and it will learn to predict the probability that a video contains a certain human action.

Lightning Flash :class:`~flash.video.VideoClassifier` and :class:`~flash.video.VideoClassificationData`
relies on `PyTorchVideo <https://pytorchvideo.readthedocs.io/en/latest/index.html>`_ internally.

You can use any models from `PyTorchVideo Model Zoo <https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html>`_
with the :class:`~flash.video.VideoClassifier`.

------

**********
Finetuning
**********

The :class:`~flash.video.VideoClassifier` provides several pre-trained model.

.. code-block:: python

    import sys

    import torch
    from torch.utils.data import SequentialSampler

    import flash
    from flash.data.utils import download_data
    from flash.video import VideoClassificationData, VideoClassifier
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

    # 1. Download a video dataset
    download_data("PATH_OR_URL_TO_DATA")

    # 2. [Optional] Specify transforms to be used during training.
    train_transform = {
        "post_tensor_transform": Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(8),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                ]),
            ),
        ]),
        "per_batch_transform_on_device": Compose([
            ApplyTransformToKey(
                key="video",
                transform=K.VideoSequential(
                    K.Normalize(
                        torch.tensor([0.45, 0.45, 0.45]),
                        torch.tensor([0.225, 0.225, 0.225])),
                    K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
                    data_format="BCTHW",
                    same_on_frame=False
                )
            ),
        ]),
    }

    # 3. Load the data
    datamodule = VideoClassificationData.from_paths(
        train_data_path="path_to_train_data",
        clip_sampler="uniform",
        clip_duration=2,
        video_sampler=SequentialSampler,
        decode_audio=False,
        train_transform=train_transform
    )

    # 4. List the available models
    print(VideoClassifier.available_models())

    # 5. Build the model
    model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False)

    # 6. Train the model
    trainer = flash.Trainer(fast_dev_run=True)

    # 6. Finetune the model
    trainer.finetune(model, datamodule=datamodule)


For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

Lets say you wanted to develope a model that could determine whether an image contains **ants** or **bees**, using the hymenoptera dataset.
Once we download the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.video.ImageClassificationData`.

.. note:: The dataset contains ``train`` and ``validation`` folders, and then each folder contains a **bees** folder, with pictures of bees, and an **ants** folder with images of, you guessed it, ants.

.. code-block::

    video_dataset
    ├── train
    │   ├── class_1
    │   │   ├── a.ext
    │   │   ├── b.ext
    │   │   ...
    │   └── class_n
    │       ├── c.ext
    │       ├── d.ext
    │       ...
    └── val
        ├── class_1
        │   ├── e.ext
        │   ├── f.ext
        │   ...
        └── class_n
            ├── g.ext
            ├── h.ext
            ...


.. code-block:: python

    import sys

    import torch
    from torch.utils.data import SequentialSampler

    import flash
    from flash.data.utils import download_data
    from flash.video import VideoClassificationData, VideoClassifier
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

    # 1. Download a video dataset
    download_data("NEED_TO_BE_CREATED")

    # 2. [Optional] Specify transforms to be used during training.
    train_transform = {
        "post_tensor_transform": Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(8),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                ]),
            ),
        ]),
        "per_batch_transform_on_device": Compose([
            ApplyTransformToKey(
                key="video",
                transform=K.VideoSequential(
                    K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                    K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
                    data_format="BCTHW",
                    same_on_frame=False
                )
            ),
        ]),
    }

    # 3. Load the data
    datamodule = VideoClassificationData.from_paths(
        train_folder="path_to_train_data",
        clip_sampler="uniform",
        clip_duration=2,
        video_sampler=SequentialSampler,
        decode_audio=False,
        train_transform=train_transform
    )

    # 4. List the available models
    print(VideoClassifier.available_models())

    # 5. Build the model
    model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False)

    # 6. Train the model
    trainer = flash.Trainer(fast_dev_run=True)

    # 6. Finetune the model
    trainer.finetune(model, datamodule=datamodule)

------

*************
API reference
*************

.. _video_classifier:

VideoClassifier
---------------

.. autoclass:: flash.video.VideoClassifier
    :members:
    :exclude-members: forward

.. _video_classification_data:

VideoClassificationData
-----------------------

.. autoclass:: flash.video.VideoClassificationData

.. automethod:: flash.video.VideoClassificationData.from_paths
