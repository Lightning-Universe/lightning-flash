To train a task from scratch:

1. Load your data and organize it using a DataModule customized for the task (example: :class:`~flash.image.ImageClassificationData`).
2. Choose and initialize your Task (setting ``pretrained=False``) which has state-of-the-art backbones built in (example: :class:`~flash.image.ImageClassifier`).
3. Init a :class:`flash.core.trainer.Trainer` or a :class:`pytorch_lightning.trainer.Trainer`.
4. Call :func:`flash.core.trainer.Trainer.fit` with your data set.
5. Save your trained model.

|

Here's an example:

.. testcode:: training

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
    model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, pretrained=False)

    # 3. Create the trainer (run one epoch for demo)
    trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())

    # 4. Train the model
    trainer.fit(model, datamodule=datamodule)

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")

.. testoutput:: training
    :hide:

    ...
