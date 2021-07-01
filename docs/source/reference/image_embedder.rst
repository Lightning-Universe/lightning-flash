
.. _image_embedder:

##############
Image Embedder
##############

********
The Task
********

Image embedding encodes an image into a vector of features which can be used for a downstream task.
This could include: clustering, similarity search, or classification.

------

*******
Example
*******

In this example we'll show how to use the :class:`~flash.image.ImageEmbedder` with a pretrained backbone to obtain feature vectors from the hymenoptera data.
Once we've downloaded the data, we create the :class:`~flash.image.ImageEmbedder` and perform inference using :meth:`~flash.image.ImageEmbedder.predict`.
Here's the full example:

.. literalinclude:: ../../../flash_examples/image_embedder.py
    :language: python
    :lines: 14-
