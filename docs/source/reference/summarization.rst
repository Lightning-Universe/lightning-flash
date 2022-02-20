.. customcarditem::
   :header: Summarization
   :card_description: Learn to summarize long passages of text with Flash and build an example model with the XSUM data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text,Summarization,NLP

.. _summarization:

#############
Summarization
#############

********
The Task
********

Summarization is the task of summarizing text from a larger document/article into a short sentence/description.
For example, taking a web article and describing the topic in a short sentence.
This task is a subset of `Sequence to Sequence tasks <https://paperswithcode.com/method/seq2seq>`_, which require the model to generate a variable length sequence given an input sequence.
In our case the article would be our input sequence, and the short description/sentence would be the output sequence from the model.

-----

*******
Example
*******

Let's look at an example.
We'll use the XSUM dataset, which contains a ``train.csv`` and ``valid.csv``.
Each CSV file looks like this:

.. code-block::

    input,target
    "The researchers have sequenced the genome of a strain of bacterium that causes the virulent infection...","A team of UK scientists hopes to shed light on the mysteries of bleeding canker, a disease that is threatening the nation's horse chestnut trees."
    "Knight was shot in the leg by an unknown gunman at Miami's Shore Club where West was holding a pre-MTV Awards...",Hip hop star Kanye West is being sued by Death Row Records founder Suge Knight over a shooting at a beach party in August 2005.
    ...

In the above, the input column represents the long articles/documents, and the target is the short description used as the target.
Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.text.seq2seq.summarization.data.SummarizationData`.
We select a pre-trained backbone to use for our :class:`~flash.text.seq2seq.summarization.model.SummarizationTask` and finetune on the XSUM data.
The backbone can be any Seq2Seq summarization model from `HuggingFace/transformers <https://huggingface.co/models?filter=pytorch&pipeline_tag=summarization>`_.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.text.seq2seq.summarization.data.SummarizationData` and the :class:`~flash.text.seq2seq.summarization.model.SummarizationTask`!

Next, we use the trained :class:`~flash.text.seq2seq.summarization.model.SummarizationTask` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/summarization.py
    :language: python
    :lines: 14-

To learn more about available for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The summarization task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash summarization

To view configuration options and options for running the summarization task with your own data, use:

.. code-block:: bash

    flash summarization --help

------

*******
Serving
*******

The :class:`~flash.text.seq2seq.summarization.model.SummarizationTask` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/summarization/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/summarization/client.py
    :language: python
    :lines: 14-

------

.. _summarization_ort:

**********************************************
Accelerate Training & Inference with Torch ORT
**********************************************

`Torch ORT <https://cloudblogs.microsoft.com/opensource/2021/07/13/accelerate-pytorch-training-with-torch-ort/>`__ converts your model into an optimized ONNX graph, speeding up training & inference when using NVIDIA or AMD GPUs. Enabling Torch ORT requires a single flag passed to the ``SummarizationTask`` once installed. See installation instructions `here <https://github.com/pytorch/ort#install-in-a-local-python-environment>`__.

.. note::

    Not all Transformer models are supported. See `this table <https://github.com/microsoft/onnxruntime-training-examples#examples>`__ for supported models + branches containing fixes for certain models.

.. code-block:: python

    ...

    model = SummarizationTask(backbone="t5-large", num_classes=datamodule.num_classes, enable_ort=True)
