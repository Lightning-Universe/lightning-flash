
.. _multi_label_classification:

################################
Multi-label Image Classification
################################

********
The task
********
Multi-label classification is the task of assigning a number of labels from a fixed set to each data point, which can be in any modality. In this example, we will look at the task of trying to predict the movie genres from an image of the movie poster.

------

********
The data
********
The data we will use in this example is a subset of the awesome movie poster genre prediction data set from the paper "Movie Genre Classification based on Poster Images with Deep Neural Networks" by Wei-Ta Chu and Hung-Jui Guo, resized to 128 by 128.
Take a look at their paper (and please consider citing their paper if you use the data) here: `www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/ <https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/>`_.

------

*********
Inference
*********

The :class:`~flash.image.ImageClassifier` is already pre-trained on `ImageNet <http://www.image-net.org/>`_, a dataset of over 14 million images.

We can use the :class:`~flash.image.ImageClassifier` model (pretrained on our data) for inference on any string sequence using :func:`~flash.image.ImageClassifier.predict`.
We can also add a simple visualisation by extending :class:`~flash.core.data.base_viz.BaseVisualization`, like this:

.. literalinclude:: ../../../flash_examples/predict/image_classification_multi_label.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

Now let's look at how we can finetune a model on the movie poster data.
Once we download the data using :func:`~flash.core.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.image.ImageClassificationData`.

.. note:: The dataset contains ``train`` and ``validation`` folders, and then each folder contains images and a ``metadata.csv`` which stores the labels.

.. code-block::

    movie_posters
    ├── train
    │   ├── metadata.csv
    │   ├── tt0084058.jpg
    │   ├── tt0084867.jpg
    │   ...
    └── val
        ├── metadata.csv
        ├── tt0200465.jpg
        ├── tt0326965.jpg
        ...


The ``metadata.csv`` files in each folder contain our labels, so we need to create a function (``load_data``) to extract the list of images and associated labels:

.. literalinclude:: ../../../flash_examples/finetuning/image_classification_multi_label.py
    :language: python
    :lines: 14-

------

For more backbone options, see :ref:`image_classification`.
