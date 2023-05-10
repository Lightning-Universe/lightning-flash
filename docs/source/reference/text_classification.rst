.. customcarditem::
   :header: Text Classification
   :card_description: Learn to classify text with Flash and build an example sentiment analyser for IMDB reviews.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/text_classification.svg
   :tags: Text,Classification,NLP

.. _text_classification:

###################
Text Classification
###################

********
The Task
********

Text classification is the task of assigning a piece of text (word, sentence or document) an appropriate class, or category.
The categories depend on the chosen data set and can range from topics.

-----

*******
Example
*******

Let's train a model to classify text as expressing either positive or negative sentiment.
We will be using the IMDB data set, that contains a ``train.csv`` and ``valid.csv``.
Here's the structure:

.. code-block::

    review,sentiment
    "Japanese indie film with humor ... ",positive
    "Isaac Florentine has made some ...",negative
    "After seeing the low-budget ...",negative
    "I've seen the original English version ...",positive
    "Hunters chase what they think is a man through ...",negative
    ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.text.classification.data.TextClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.text.classification.model.TextClassifier` and finetune on the IMDB data.
The backbone can be any BERT classification model from `HuggingFace/transformers <https://huggingface.co/models?filter=pytorch&pipeline_tag=text-classification>`_.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.text.classification.model.TextClassifier` and the :class:`~flash.text.classification.data.TextClassificationData`!

Next, we use the trained :class:`~flash.text.classification.model.TextClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../examples/text/text_classification.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The text classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash text_classification

To view configuration options and options for running the text classifier with your own data, use:

.. code-block:: bash

    flash text_classification --help

------

*******
Serving
*******

The :class:`~flash.text.classification.model.TextClassifier` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../examples/serve/text_classification/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../examples/serve/text_classification/client.py
    :language: python
    :lines: 14-

------

.. _text_classification_ort:

**********************************************
Accelerate Training & Inference with Torch ORT
**********************************************

`Torch ORT <https://cloudblogs.microsoft.com/opensource/2021/07/13/accelerate-pytorch-training-with-torch-ort/>`__ converts your model into an optimized ONNX graph, speeding up training & inference when using NVIDIA or AMD GPUs. Enabling Torch ORT requires a single flag passed to the ``TextClassifier`` once installed. See installation instructions `here <https://github.com/pytorch/ort#install-in-a-local-python-environment>`__.

.. note::

    Not all Transformer models are supported. See `this table <https://github.com/microsoft/onnxruntime-training-examples#examples>`__ for supported models + branches containing fixes for certain models.

.. code-block:: python

    ...

    model = TextClassifier(backbone="facebook/bart-large", num_classes=datamodule.num_classes, enable_ort=True)
