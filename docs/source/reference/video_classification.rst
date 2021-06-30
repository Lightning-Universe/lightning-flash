
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
Once we download the data using :func:`~flash.core.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.video.VideoClassificationData`.

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


.. literalinclude:: ../../../flash_examples/finetuning/video_classification.py
    :language: python
    :lines: 14-
