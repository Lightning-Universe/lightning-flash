.. _summarization:

#############
Summarization
#############

********
The task
********

Summarization is the task of summarizing text from a larger document/article into a short sentence/description. For example, taking a web article and describing the topic in a short sentence.
This task is a subset of `Sequence to Sequence tasks <https://paperswithcode.com/method/seq2seq>`_, which requires the model to generate a variable length sequence given an input sequence. In our case the article would be our input sequence, and the short description/sentence would be the output sequence from the model.

-----

*********
Inference
*********

The :class:`~flash.text.SummarizationTask` is already pre-trained on `XSUM <https://arxiv.org/abs/1808.08745>`_, a dataset of online British Broadcasting Corporation articles.

Use the :class:`~flash.text.SummarizationTask` pretrained model for inference on any string sequence using :class:`~flash.text.SummarizationTask` `predict` method:

.. code-block:: python

    # import our libraries
    from flash.text import SummarizationTask

    # 1. Load the model from a checkpoint
    model = SummarizationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/summarization_model_xsum.pt")

    # 2. Perform inference from a sequence
    predictions = model.predict([
        """
        Camilla bought a box of mangoes with a Brixton Â£10 note, introduced last year to try to keep the money of local
        people within the community.The couple were surrounded by shoppers as they walked along Electric Avenue.
        They came to Brixton to see work which has started to revitalise the borough.
        It was Charles' first visit to the area since 1996, when he was accompanied by the former
        South African president Nelson Mandela.Greengrocer Derek Chong, who has run a stall on Electric Avenue
        for 20 years, said Camilla had been ""nice and pleasant"" when she purchased the fruit.
        ""She asked me what was nice, what would I recommend, and I said we've got some nice mangoes.
        She asked me were they ripe and I said yes - they're from the Dominican Republic.""
        Mr Chong is one of 170 local retailers who accept the Brixton Pound.
        Customers exchange traditional pound coins for Brixton Pounds and then spend them at the market
        or in participating shops.
        During the visit, Prince Charles spent time talking to youth worker Marcus West, who works with children
        nearby on an estate off Coldharbour Lane. Mr West said:
        ""He's on the level, really down-to-earth. They were very cheery. The prince is a lovely man.""
        He added: ""I told him I was working with young kids and he said, 'Keep up all the good work.'""
        Prince Charles also visited the Railway Hotel, at the invitation of his charity The Prince's Regeneration Trust.
        The trust hopes to restore and refurbish the building,
        where once Jimi Hendrix and The Clash played, as a new community and business centre."
        """
    ])
    print(predictions)

Or on a given dataset, use :class:`~flash.core.trainer.Trainer` `predict` method:

.. code-block:: python

    # import our libraries
    from flash import Trainer
    from flash.data.utils import download_data
    from flash.text import SummarizationData, SummarizationTask

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = SummarizationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/summarization_model_xsum.pt")

    # 3. Create dataset from file
    datamodule = SummarizationData.from_file(
        predict_file="data/xsum/predict.csv",
        input="input",
    )

    # 4. generate summaries
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)

For more advanced inference options, see :ref:`predictions`.

-----

**********
Finetuning
**********

Say you want to finetune to your own summarization data. We use the XSUM dataset as an example which contains a ``train.csv`` and ``valid.csv``, structured like so:

.. code-block::

    input,target
    "The researchers have sequenced the genome of a strain of bacterium that causes the virulent infection...","A team of UK scientists hopes to shed light on the mysteries of bleeding canker, a disease that is threatening the nation's horse chestnut trees."
    "Knight was shot in the leg by an unknown gunman at Miami's Shore Club where West was holding a pre-MTV Awards...",Hip hop star Kanye West is being sued by Death Row Records founder Suge Knight over a shooting at a beach party in August 2005.
    ...

In the above the input column represents the long articles/documents, and the target is the short description used as the target.

All we need is three lines of code to train our model!

.. code-block:: python

    # import our libraries
    import flash
    from flash.data.utils import download_data
    from flash.text import SummarizationData, SummarizationTask

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", 'data/')

    # Organize the data
    datamodule = SummarizationData.from_files(
        train_file="data/xsum/train.csv",
        valid_file="data/xsum/valid.csv",
        test_file="data/xsum/test.csv",
        input="input",
        target="target"
    )

    # 2. Build the task
    model = SummarizationTask()

    # 4. Create trainer
    trainer = flash.Trainer(max_epochs=1, gpus=1)

    # 5. Finetune the task
    trainer.finetune(model, datamodule=datamodule)

    # 6. Save trainer task
    trainer.save_checkpoint("summarization_model_xsum.pt")

----

To run the example:

.. code-block:: bash

    python flash_examples/finetuning/summarization.py


------

*********************
Changing the backbone
*********************
By default, we use the `t5 <https://arxiv.org/abs/1910.10683>`_ model for summarization. You can change the model run by the task to any summarization model from `HuggingFace/transformers <https://huggingface.co/models?filter=summarization,pytorch>`_ by passing in a backbone parameter.

.. note:: When changing the backbone, make sure you pass in the same backbone to the Task and the Data object! Since this is a Seq2Seq task, make sure you use a Seq2Seq model.

.. code-block:: python

    # use google/mt5-small, covering 101 languages
    datamodule = SummarizationData.from_files(
        backbone="google/mt5-small",
        train_file="data/wmt_en_ro/train.csv",
        valid_file="data/wmt_en_ro/valid.csv",
        test_file="data/wmt_en_ro/test.csv",
        input="input",
        target="target",
    )

    model = SummarizationTask(backbone="google/mt5-small")

------

*************
API reference
*************

.. _summarization_task:

SummarizationTask
-----------------

.. autoclass:: flash.text.SummarizationTask
    :members:
    :exclude-members: forward

.. _summarization_data:

SummarizationData
-----------------

.. autoclass:: flash.text.SummarizationData

.. automethod:: flash.text.SummarizationData.from_files
