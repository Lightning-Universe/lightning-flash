.. customcarditem::
   :header: Extractive Question Answering
   :card_description: Learn to answer questions pertaining to some known textual context.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/extractive_question_answering.svg
   :tags: NLP,Text

.. _question_answering:

##################
Question Answering
##################

********
The Task
********

Question Answering is the task of being able to answer questions pertaining to some known context.
For example, given a context about some historical figure, any question pertaininig to the context should be answerable.
In our case the article would be our input context and question, and the answer would be the output sequence from the model.

.. note::

    We currently only support Extractive Question Answering, like the task performed using the SQUAD like datasets.

-----

*******
Example
*******

Let's look at an example.
We'll use the SQUAD 2.0 dataset, which contains ``train-v2.0.json`` and ``dev-v2.0.json``.
Each JSON file looks like this:

.. code-block::

    {
		"answers": {
			"answer_start": [94, 87, 94, 94],
			"text": ["10th and 11th centuries", "in the 10th and 11th centuries", "10th and 11th centuries", "10th and 11th centuries"]
		},
		"context": "\"The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave thei...",
		"id": "56ddde6b9a695914005b9629",
		"question": "When were the Normans in Normandy?",
		"title": "Normans"
	}
    ...

In the above, the ``context`` key represents the context used for the question and answer, the ``question`` key represents the question being asked with respect to the context, the ``answer`` key stores the answer(s) for the question.
``id`` and ``title`` are used for unique identification and grouping concepts together respectively.
Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.text.question_answering.data.QuestionAnsweringData`.
We select a pre-trained backbone to use for our :class:`~flash.text.question_answering.model.QuestionAnsweringTask` and finetune on the SQUAD 2.0 data.
The backbone can be any Question Answering model from `HuggingFace/transformers <https://huggingface.co/transformers/model_doc/auto.html#automodelforquestionanswering>`_.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.text.question_answering.data.QuestionAnsweringData` and the :class:`~flash.text.question_answering.model.QuestionAnsweringTask`!

Next, we use the trained :class:`~flash.text.question_answering.model.QuestionAnsweringTask` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/question_answering.py
    :language: python
    :lines: 14-

To learn how to the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********************************************
Accelerate Training & Inference with Torch ORT
**********************************************

`Torch ORT <https://cloudblogs.microsoft.com/opensource/2021/07/13/accelerate-pytorch-training-with-torch-ort/>`__ converts your model into an optimized ONNX graph, speeding up training & inference when using NVIDIA or AMD GPUs. Enabling Torch ORT requires a single flag passed to the ``QuestionAnsweringTask`` once installed. See installation instructions `here <https://github.com/pytorch/ort#install-in-a-local-python-environment>`__.

.. note::

    Not all Transformer models are supported. See `this table <https://github.com/microsoft/onnxruntime-training-examples#examples>`__ for supported models + branches containing fixes for certain models.

.. code-block:: python

    ...

    model = QuestionAnsweringTask(backbone="distilbert-base-uncased", max_answer_length=30, enable_ort=True)
