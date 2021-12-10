.. _text_embedder:

##############
Text Embedder
##############

********
The Task
********
This task consists of creating a Sentence Embedding. That is, a vector of sentence representations which can be used for a downstream task.
The  :class:`~flash.text.TextEmbedder` implementation is adopted from `https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py`
------

*******
Example
*******

Let's look at an example of generating sentence embeddings. 

We start by using few custom sentences as input.
Next, we load a trained model from :class:`~flash.text.TextEmbedder`.
And generate sentence embeddings.
Here's the full example:

.. literalinclude:: ../../../flash_examples/text_embedder.py
    :language: python
    :lines: 14
