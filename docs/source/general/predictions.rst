
.. _predictions:

#######################
Predictions (inference)
#######################

You can use Flash to get predictions on pretrained or finetuned models.

First create a :class:`~flash.core.data.data_module.DataModule` with some predict data, then pass it to the :meth:`Trainer.predict <flash.core.trainer.Trainer.predict>` method.

.. code-block:: python

    from flash import Trainer
    from flash.core.data.utils import download_data
    from flash.image import ImageClassifier, ImageClassificationData

    # 1. Download the data set
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    # 2. Load the model from a checkpoint
    model = ImageClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/0.7.0/image_classification_model.pt"
    )

    # 3. Predict whether the image contains an ant or a bee
    trainer = Trainer()
    datamodule = ImageClassificationData.from_files(
        predict_files=["data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg"]
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)
    # out: [["bees"]]


Serializing predictions
=======================

To change the output format of predictions you can attach an :class:`~flash.core.data.io.output.Output` to your
:class:`~flash.core.model.Task`. For example, you can choose to output probabilities (for more options see the API
reference below).

.. code-block:: python

    from flash.core.classification import ProbabilitiesOutput
    from flash.core.data.utils import download_data
    from flash.image import ImageClassifier


    # 1. Download the data set
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    # 2. Load the model from a checkpoint
    model = ImageClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/0.7.0/image_classification_model.pt"
    )

    # 3. Attach the Output
    model.output = ProbabilitiesOutput()

    # 4. Predict whether the image contains an ant or a bee
    trainer = Trainer()
    datamodule = ImageClassificationData.from_files(
        predict_files=["data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg"]
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)
    # out: [[[0.5926494598388672, 0.40735048055648804]]]


.. note::
    PyTorch Lightning does not return predictions directly from `predict` when using a multi-GPU configuration (DDP). Instead you should use a :class:`pytorch_lightning.callbacks.BasePredictionWriter`.
