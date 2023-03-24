.. customcarditem::
   :header: Style Transfer
   :card_description: Learn about image style transfer with Flash and build an example which transfers style from The Starry Night to images from the COCO data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/style_transfer.svg
   :tags: Image,Style-Transfer
   :beta:

.. beta:: Style transfer is currently in Beta.

.. _style_transfer:

##############
Style Transfer
##############

********
The Task
********

The Neural Style Transfer Task is an optimization method which extract the style from an image and apply it another image while preserving its content.
The goal is that the output image looks like the content image, but “painted” in the style of the style reference image.

.. image:: https://raw.githubusercontent.com/pystiche/pystiche/main/docs/source/graphics/banner/banner.jpg
    :alt: style_transfer_example

The :class:`~flash.image.style_transfer.model.StyleTransfer` and :class:`~flash.image.style_transfer.data.StyleTransferData` classes internally rely on `pystiche <https://pystiche.org>`_.

------

*******
Example
*******

Let's look at transferring the style from `The Starry Night <https://en.wikipedia.org/wiki/The_Starry_Night>`_ onto the images from the COCO 128 data set from the :ref:`object_detection` Guide.
Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.style_transfer.data.StyleTransferData`.
Next, we create our :class:`~flash.image.style_transfer.model.StyleTransfer` task with the desired style image and fit on the COCO 128 images.
We then use the trained :class:`~flash.image.style_transfer.model.StyleTransfer` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../examples/style_transfer.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The style transfer task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash style_transfer

To view configuration options and options for running the style transfer task with your own data, use:

.. code-block:: bash

    flash style_transfer --help
