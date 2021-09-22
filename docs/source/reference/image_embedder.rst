.. customcarditem::
   :header: Image Embedder
   :card_description: Learn to generate embeddings from images with Flash.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/image_embedder.svg
   :tags: Image,Embedding

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

Let's see how to use the :class:`~flash.image.embedding.model.ImageEmbedder` with a pretrained backbone to obtain feature vectors from the hymenoptera data.
Once we've downloaded the data, we create the :class:`~flash.image.embedding.model.ImageEmbedder` and perform inference (obtaining feature vectors / embeddings) using :meth:`~flash.image.embedding.model.ImageEmbedder.predict`.
Here's the full example:

.. literalinclude:: ../../../flash_examples/image_embedder.py
    :language: python
    :lines: 14-
