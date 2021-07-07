
.. _image_classification_multi_label:

################################
Multi-label Image Classification
################################

********
The Task
********

Multi-label classification is the task of assigning a number of labels from a fixed set to each data point, which can be in any modality (images in this case).
Multi-label image classification is supported by the :class:`~flash.image.classification.model.ImageClassifier` via the ``multi-label`` argument.

------

*******
Example
*******

Let's look at the task of trying to predict the movie genres from an image of the movie poster.
The data we will use is a subset of the awesome movie poster genre prediction data set from the paper "Movie Genre Classification based on Poster Images with Deep Neural Networks" by Wei-Ta Chu and Hung-Jui Guo, resized to 128 by 128.
Take a look at their paper (and please consider citing their paper if you use the data) here: `www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/ <https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/>`_.
The data set contains ``train`` and ``validation`` folders, and then each folder contains images and a ``metadata.csv`` which stores the labels.
Here's an overview:

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

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we need to create the :class:`~flash.image.classification.data.ImageClassificationData`.
We first create a function (``load_data``) to extract the list of images and associated labels which can then be passed to :meth:`~flash.image.classification.data.ImageClassificationData.from_files`.
We select a pre-trained backbone to use for our :class:`~flash.image.classification.model.ImageClassifier` and fine-tune on the posters data.
We then use the trained :class:`~flash.image.classification.model.ImageClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/image_classification_multi_label.py
    :language: python
    :lines: 14-

------

*******
Serving
*******

The :class:`~flash.image.classification.model.ImageClassifier` is servable.
For more information, see :ref:`image_classification`.
