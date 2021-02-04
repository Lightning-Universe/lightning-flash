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

Use the :class:`~flash.text.TranslationTask` pretrained model for inference on any string sequence using :class:`~flash.text.TranslationTask` `predict` method:

.. code-block:: python

    # import our libraries
    from flash.text import TranslationTask

    # 1. Load the model from a checkpoint
    model = TranslationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/translation_model_en_ro.pt")

    # 2. Perform inference from list of sequences
    predictions = model.predict([
        "BBC News went to meet one of the project's first graduates.",
        "A recession has come as quickly as 11 months after the first rate hike and as long as 86 months.",
    ])
    print(predictions)

Or on a given dataset, use :class:`~flash.core.trainer.Trainer` `predict` method:

.. code-block:: python

    # import our libraries
    from flash import Trainer
    from flash import download_data
    from flash.text import TranslationData, TranslationTask

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = TranslationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/translation_model_en_ro.pt")

    # 3. Create dataset from file
    datamodule = TranslationData.from_file(
        predict_file="data/wmt_en_ro/predict.csv",
        input="input",
    )

    # 4. generate translations
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)

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

.. code-block:: python

    # import our libraries
    import flash
    from flash import download_data
    from flash.text import TranslationData, TranslationTask

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", 'data/')

    # Organize the data
    datamodule = TranslationData.from_files(
        train_file="data/wmt_en_ro/train.csv",
        valid_file="data/wmt_en_ro/valid.csv",
        test_file="data/wmt_en_ro/test.csv",
        input="input",
        target="target",
    )

    # 2. Build the task
    model = TranslationTask()

    # 4. Create trainer- in this case we need to run on gpus, `precision=16` boosts speed
    trainer = flash.Trainer(max_epochs=5, gpus=1, precision=16)

    # 5. Finetune the task
    trainer.finetune(model, datamodule=datamodule)

    # 6. Save model to checkpoint
    trainer.save_checkpoint("translation_model_en_ro.pt")

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

.. code-block:: python

    datamodule = TranslationData.from_files(
        train_file="data/wmt_en_ro/train.csv",
        valid_file="data/wmt_en_ro/valid.csv",
        test_file="data/wmt_en_ro/test.csv",
        input="input",
        target="target",
        backbone="t5-small",
    )


------

*************
API reference
*************

.. _translation_task:

TranslationTask
---------------

.. autoclass:: flash.text.TranslationTask
    :members:
    :exclude-members: forward

.. _translation_data:

TranslationData
---------------

.. autoclass:: flash.text.TranslationData

.. automethod:: flash.text.TranslationData.from_files
