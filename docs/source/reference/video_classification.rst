
.. _video_classification:

####################
Video Classification
####################

********
The task
********

Typically, Video Classification refers to the task of producing a label for actions identified in a given video.

The task predicts which ‘class’ the video clip most likely belongs to with a degree of certainty.

A class is a label that describes what action is being performed within the video clip, such as **swimming** , **playing piano**, etc.

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

Let's say you wanted to develop a model that could determine whether a video clip contains a human **swimming** or **playing piano**,
using the `Kinetics dataset <https://deepmind.com/research/open-source/kinetics>`_.
Once we download the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.video.VideoClassificationData`.

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

    # 1. Download a video clip dataset. Find more dataset at https://pytorchvideo.readthedocs.io/en/latest/data.html
    download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip")

    # 2. [Optional] Specify transforms to be used during training.
    # Flash helps you to place your transform exactly where you want.
    # Learn more at https://lightning-flash.readthedocs.io/en/latest/general/data.html#flash.data.process.Preprocess
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

    # 3. Load the data from directories.
    datamodule = VideoClassificationData.from_paths(
        train_data_path="data/kinetics/train",
        val_data_path="data/kinetics/val",
        predict_data_path="data/kinetics/predict",
        clip_sampler="uniform",
        clip_duration=2,
        video_sampler=SequentialSampler,
        decode_audio=False,
        train_transform=train_transform
    )

    # 4. List the available models
    print(VideoClassifier.available_models())
    # out: ['efficient_x3d_s', 'efficient_x3d_xs', ... ,slowfast_r50', 'x3d_m', 'x3d_s', 'x3d_xs']

    # 5. Build the model
    model = VideoClassifier(model="x3d_xs", num_classes=datamodule.num_classes, pretrained=False)

    # 6. Train the model
    trainer = flash.Trainer(fast_dev_run=True)

    # 6. Finetune the model
    trainer.finetune(model, datamodule=datamodule)

    predictions = model.predict("data/kinetics/train/archery/-1q7jA3DXQM_000005_000015.mp4")
    print(predictions)


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
