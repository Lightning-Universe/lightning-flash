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
