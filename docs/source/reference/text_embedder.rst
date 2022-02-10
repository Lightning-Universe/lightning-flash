.. customcarditem::
   :header: Text Embedder
   :card_description: Learn to generate sentence embeddings with Flash.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/multi_label_text_classification.svg
   :tags: Text,Embedding

.. _text_embedder:

#############
Text Embedder
#############

********
The Task
********

This task consists of creating a Sentence Embedding. That is, a vector of sentence representations which can be used for a downstream task.
The  :class:`~flash.text.embedding.model.TextEmbedder` implementation relies on components from `sentence-transformers <https://github.com/UKPLab/sentence-transformers>`_.

------

*******
Example
*******

Let's look at an example of generating sentence embeddings.

We start by loading some sentences for prediction with the :class:`~flash.text.classification.data.TextClassificationData` class.
Next, we create our :class:`~flash.text.embedding.model.TextEmbedder` with a pretrained backbone from the `HuggingFace hub <https://huggingface.co/models?filter=pytorch>`_.
Finally, we create a :class:`~flash.core.trainer.Trainer` and generate sentence embeddings.
Here's the full example:

.. literalinclude:: ../../../flash_examples/text_embedder.py
    :language: python
    :lines: 14-
