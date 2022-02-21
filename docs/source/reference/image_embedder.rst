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

The :class:`~flash.image.embedding.model.ImageEmbedder` internally relies on `VISSL <https://vissl.ai/>`_.

------

*******
Example
*******

Let's see how to configure a training strategy for the :class:`~flash.image.embedding.model.ImageEmbedder` task.
A vanilla :class:`~flash.core.data.data_module.DataModule` object be created using standard Datasets as shown below.
Then the user can configure the :class:`~flash.image.embedding.model.ImageEmbedder` task with ``training_strategy``, ``backbone``, ``head`` and ``pretraining_transform``.
There are options provided to send additional arguments to config selections.
This task can now be sent to the ``fit()`` method of :class:`~flash.core.trainer.Trainer`.

.. note::

   A lot of VISSL loss functions use hard-coded ``torch.distributed`` methods. The user is suggested to use ``accelerator=ddp`` even with a single GPU.
   Only ``barlow_twins`` training strategy works on the CPU. All other loss functions are configured to work on GPUs.

.. literalinclude:: ../../../flash_examples/image_embedder.py
    :language: python
    :lines: 14-

To learn how to the available backbones / heads for this task, see :ref:`backbones_heads`.
