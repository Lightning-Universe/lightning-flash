.. _text_classification_multi_label:

###############################
Multi-label Text Classification
###############################

********
The Task
********

Multi-label classification is the task of assigning a number of labels from a fixed set to each data point, which can be in any modality (text in this case).
Multi-label text classification is supported by the :class:`~flash.text.TextClassifier` via the ``multi-label`` argument.

-----

********
The data
********

------

*******
Example
*******

Let's look at the task of classifying comment toxicity.
The data we will use in this example is from the kaggle toxic comment classification challenge by jigsaw: `www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>`_.
The data is stored in CSV files with this structure:

.. code-block::

    "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    "0000997932d777bf","...",0,0,0,0,0,0
    "0002bcb3da6cb337","...",1,1,1,0,1,0
    "0005c987bdfc9d4b","...",1,0,0,0,0,0
    ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.text.TextClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.text.TextClassifier` and finetune on the toxic comments data.
The backbone can be any BERT classification model from `HuggingFace/transformers <https://huggingface.co/models?filter=pytorch&pipeline_tag=text-classification>`_.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.text.TextClassifier` and the :class:`~flash.text.TextClassificationData`!

Next, we use the trained :class:`~flash.text.TextClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/text_classification_multi_label.py
    :language: python
    :lines: 14-

------

*******
Serving
*******

The :class:`~flash.text.TextClassifier` is servable.
For more information, see :ref:`text_classification`.
