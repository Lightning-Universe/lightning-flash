.. customcarditem::
   :header: Image Embedder
   :card_description: Learn to generate embeddings from images with Flash.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/image_embedder.svg
   :tags: Image,Embedding
   :beta:

.. beta:: The VISSL integration is currently in Beta.

.. warning::

   Multi-gpu training is not currently supported by the :class:`~flash.image.embedding.model.ImageEmbedder` task.

.. _image_embedder:

##############
Image Embedder
##############

********
The Task
********

Image embedding encodes an image into a vector of features which can be used for a downstream task.
This could include: clustering, similarity search, or classification.

The Flash :class:`~flash.image.embedding.model.ImageEmbedder` can be trained with Self Supervised Learning (SSL) to improve the quality of the embeddings it produces for your data.
The :class:`~flash.image.embedding.model.ImageEmbedder` internally relies on `VISSL <https://vissl.ai/>`_.
You can read more about our integration with VISSL here: :ref:`vissl`.

------

*******
Example
*******

Let's see how to configure a training strategy for the :class:`~flash.image.embedding.model.ImageEmbedder` task.
First we create an :class:`~flash.image.classification.data.ImageClassificationData` object using a `Dataset` from torchvision.
Next, we configure the :class:`~flash.image.embedding.model.ImageEmbedder` task with ``training_strategy``, ``backbone``, ``head`` and ``pretraining_transform``.
Finally, we construct a :class:`~flash.core.trainer.Trainer` and call ``fit()``.
Here's the full example:

.. literalinclude:: ../../../examples/image/image_embedder.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.
You can view the available training strategies with the :meth:`~flash.image.embedding.model.ImageEmbedder.available_training_strategies` method.

The ``head`` and ``pretraining_transform`` arguments should match the choice of ``training_strategy`` following this table:

=====================  =====================  ==========================
``training_strategy``  ``head``               ``pretraining_transform``
=====================  =====================  ==========================
``simclr``             ``simclr_head``        ``simclr_transform``
``barlow_twins``       ``barlow_twins_head``  ``barlow_twins_transform``
``swav``               ``swav_head``          ``swav_transform``
=====================  =====================  ==========================
