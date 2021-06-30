.. _text_classification_multi_label:

###############################
Multi-label Text Classification
###############################

********
The task
********

Multi-label classification is the task of assigning a number of labels from a fixed set to each data point, which can be in any modality.
In this example, we will look at the task of classifying comment toxicity.

-----

********
The data
********
The data we will use in this example is from the kaggle toxic comment classification challenge by jigsaw: `www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>`_.

------

*********
Inference
*********

We can load a pretrained :class:`~flash.text.classification.model.TextClassifier` and perform inference on any string sequence using :func:`~flash.text.classification.model.TextClassifier.predict`:

.. literalinclude:: ../../../flash_examples/predict/text_classification_multi_label.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

-----

**********
Finetuning
**********

Now let's look at how we can finetune a model on the toxic comments data.
Once we download the data using :func:`~flash.core.data.download_data`, we can create our :meth:`~flash.text.classification.data.TextClassificationData` using :meth:`~flash.core.data.data_module.DataModule.from_csv`.
The backbone can be any BERT classification model from Huggingface.
We use ``"unitary/toxic-bert"`` as the backbone since it's already trained on the toxic comments data.
Now all we need to do is fine-tune our model!

.. literalinclude:: ../../../flash_examples/finetuning/text_classification_multi_label.py
    :language: python
    :lines: 14-

----

To run the example:

.. code-block:: bash

    python flash_examples/finetuning/text_classification_multi_label.py
