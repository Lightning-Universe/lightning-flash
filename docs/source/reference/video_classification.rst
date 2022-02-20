.. customcarditem::
   :header: Video Classification
   :card_description: Learn to classify videos with Flash and build an example action classifier.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/video_classification.svg
   :tags: Video,Classification

.. _video_classification:

####################
Video Classification
####################

********
The Task
********

Typically, Video Classification refers to the task of producing a label for actions identified in a given video.
The task is to predict which *class* the video clip belongs to.

Lightning Flash :class:`~flash.video.classification.model.VideoClassifier` and :class:`~flash.video.classification.data.VideoClassificationData` classes internally rely on `PyTorchVideo <https://pytorchvideo.readthedocs.io/en/latest/index.html>`_.

------

*******
Example
*******

Let's develop a model to classifying video clips of Humans performing actions (such as: **archery** , **bowling**, etc.).
We'll use data from the `Kinetics dataset <https://deepmind.com/research/open-source/kinetics>`_.
Here's an outline of the folder structure:

.. code-block::

    video_dataset
    ├── train
    │   ├── archery
    │   │   ├── -1q7jA3DXQM_000005_000015.mp4
    │   │   ├── -5NN5hdIwTc_000036_000046.mp4
    │   │   ...
    │   ├── bowling
    │   │   ├── -5ExwuF5IUI_000030_000040.mp4
    │   │   ├── -7sTNNI1Bcg_000075_000085.mp4
    │   ... ...
    └── val
        ├── archery
        │   ├── 0S-P4lr_c7s_000022_000032.mp4
        │   ├── 2x1lIrgKxYo_000589_000599.mp4
        │   ...
        ├── bowling
        │   ├── 1W7HNDBA4pA_000002_000012.mp4
        │   ├── 4JxH3S5JwMs_000003_000013.mp4
        ... ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.video.classification.data.VideoClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.video.classification.model.VideoClassifier` and fine-tune on the Kinetics data.
The backbone can be any model from the `PyTorchVideo Model Zoo <https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html>`_.
We then use the trained :class:`~flash.video.classification.model.VideoClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/video_classification.py
    :language: python
    :lines: 14-

To learn more about available for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The video classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash video_classification

To view configuration options and options for running the video classifier with your own data, use:

.. code-block:: bash

    flash video_classification --help
