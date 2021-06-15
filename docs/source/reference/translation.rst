.. _translation:

###########
Translation
###########

********
The Task
********

Translation is the task of translating text from a source language to another, such as English to Romanian.
This task is a subset of `Sequence to Sequence tasks <https://paperswithcode.com/method/seq2seq>`_, which requires the model to generate a variable length sequence given an input sequence. In our case, the task will take an English sequence as input, and output the same sequence in Romanian.

-----

*********
Inference
*********

The :class:`~flash.text.TranslationTask` is already pre-trained on `WMT16 English/Romanian <https://www.statmt.org/wmt16/translation-task.html>`_, a dataset of English to Romanian samples, based on the `Europarl corpora <http://www.statmt.org/europarl/>`_.

Use the :class:`~flash.text.TranslationTask` pretrained model for inference using :class:`~flash.text.TranslationTask` `predict` method:

.. literalinclude:: ../../../flash_examples/predict/translation.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

-----

**********
Finetuning
**********

Say you want to finetune to your own translation data. We use the English/Romanian WMT16 dataset as an example which contains a ``train.csv`` and ``valid.csv``, structured like so:

.. code-block::

    input,target
    "Written statements and oral questions (tabling): see Minutes","Declaraţii scrise şi întrebări orale (depunere): consultaţi procesul-verbal"
    "Closure of sitting","Ridicarea şedinţei"
    ...

In the above the input/target columns represent the English and Romanian translation respectively.

All we need is three lines of code to train our model! By default, we use a `mBART <https://github.com/pytorch/fairseq/tree/master/examples/mbart/>`_ backbone for translation which requires a GPU to train.

.. literalinclude:: ../../../flash_examples/finetuning/translation.py
    :language: python
    :lines: 14-

----

To run the example:

.. code-block:: bash

    python flash_examples/finetuning/translation.py


------

*********************
Changing the backbone
*********************
You can change the model run by passing in the backbone parameter.

.. note:: When changing the backbone, make sure you pass in the same backbone to the Task and the Data object! Since this is a Seq2Seq task, make sure you use a Seq2Seq model.

.. testsetup::

    from flash.core.data.utils import download_data
    from flash.text import TranslationData, TranslationTask

    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", "data/")


.. testcode::

    datamodule = TranslationData.from_csv(
        "input",
        "target",
        backbone="t5-small",
        train_file="data/wmt_en_ro/train.csv",
        val_file="data/wmt_en_ro/valid.csv",
        test_file="data/wmt_en_ro/test.csv",
    )

    model = TranslationTask(backbone="t5-small")

.. testoutput::
    :hide:

    ...
