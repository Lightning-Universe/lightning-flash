To use a Task for finetuning:

1. Load your data and organize it using a DataModule customized for the task (example: :class:`~flash.image.ImageClassificationData`).
2. Choose and initialize your Task which has state-of-the-art backbones built in (example: :class:`~flash.image.ImageClassifier`).
3. Init a :class:`flash.core.trainer.Trainer`.
4. Choose a finetune strategy (example: "freeze") and call :func:`flash.core.trainer.Trainer.finetune` with your data.
5. Save your finetuned model.

|

Here's an example of finetuning.

.. testcode:: finetune

    from pytorch_lightning import seed_everything

    import flash
    from flash.core.classification import LabelsOutput
    from flash.core.data.utils import download_data
    from flash.image import ImageClassificationData, ImageClassifier

    # set the random seeds.
    seed_everything(42)

    # 1. Download and organize the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
        batch_size=1,
    )

    # 2. Build the model using desired Task
    model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

    # 3. Create the trainer (run one epoch for demo)
    trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())

    # 4. Finetune the model
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")

.. testoutput:: finetune
    :hide:

    ...


Using a finetuned model
-----------------------
Once you've finetuned, use the model to predict:

.. testcode:: finetune

    # Output predictions as labels, automatically inferred from the training data in part 2.
    model.output = LabelsOutput()

    predict_datamodule = ImageClassificationData.from_files(
        predict_files=[
            "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
            "data/hymenoptera_data/val/ants/2255445811_dabcdf7258.jpg",
        ]
    )
    predictions = trainer.predict(model, datamodule=predict_datamodule)
    print(predictions)

We get the following output:

.. testoutput:: finetune
    :hide:

    ...

.. testcode:: finetune
    :hide:

    assert all(
        [all([prediction in ["ants", "bees"] for prediction in prediction_batch]) for prediction_batch in predictions]
    )

.. code-block::

    [['bees', 'ants']]

Or you can use the saved model for prediction anywhere you want!

.. code-block:: python

    from flash import Trainer
    from flash.image import ImageClassifier, ImageClassificationData

    # load finetuned checkpoint
    model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")

    trainer = Trainer()
    datamodule = ImageClassificationData.from_files(predict_files=["path/to/your/own/image.png"])
    predictions = trainer.predict(model, datamodule=datamodule)
