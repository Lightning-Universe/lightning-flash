Quick Start
===========
Flash is a high-level deep learning framework for fast prototyping, finetuning and solving applied deep learning problems.

Flash is built on top of `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_,
a powerful deep learning research framework for training models at scale. With the power of Lightning,
you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes.

Flash is built for 3 major use-cases:

- Inference (predictions)
- Finetuning
- Training

-----

Flexibility
-----------

Unlike high-level frameworks, once you need more flexibility you can directly write a LightningModule which is just
organized PyTorch with unecessary engineering details abstracted away.

- Flash (high level)
- Lightning (mid-level)
- PyTorch (low-level)

Flash is excellent for:

- data scientists
- kagglers
- applied corporate researchers
- applied academic researchers

When you need more flexibility you can build your own tasks or simply use Lightning directly.

.. note:: Read :doc:`here <reference/flash_to_pl>` to understand when to use Flash vs Lightning.

----

Install
-------

You can install flash using pip/conda:

.. code-block:: bash

    pip install pytorch-lightning-flash

------

Tasks
-----
Flash is built as a collection of Tasks. A task is highly opinionated and laser-focused on solving a single problem
well, using state-of-the-art methods.

Here are examples of tasks:

.. code-block:: python

    from flash.text import TextClassifier
    from flash.vision import ImageClassifier
    from flash.tabular import TabularClassifier

.. note:: Tasks are inflexible by definition! For more flexibility use a LightningModule directly

Tasks are designed for:

- inference
- finetuning
- training from scratch

------

Inference
---------
Inference is the process of generating predictions. To use a task for inference:

1. Init the task.
2. Load pretrained weights.
3. Feed in the raw predictions

|

Here's an example of inference.

.. code-block:: python

    # import our libraries
    from flash.text import TextClassifier

    # Load finetuned task
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")

    # 2. Perform inference from list of sequences
    predictions = model.predict([
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado."
        "Very, very afraid"
        "This guy has done a great job with this movie!",
    ])
    print(predictions)

-------

Finetune
--------
Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset.
To use a Task for finetuning:

1. Download and set up your own data (pytorch dataloaders or lightning datamodules work).
2. Init your task.
3. Init a flash (or lightning trainer).
4. call trainer.finetune.
5. Use your model.

|

Here's an example of finetuning.

.. code-block:: python

    import flash
    from flash import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        backbone="resnet18",
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Use the model
    predictions = model.predict('data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg")
    print(predictions)

    # 7. Save the new model!
    trainer.save_checkpoint("image_classification_model.pt")

Once your model is finetuned, use it for prediction anywhere you want!

.. code-block:: python

    from flash.vision import ImageClassifier

    # load finetuned checkpoint
    model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")

    predictions = model.predict('path/to/your/own/image.png")

----

Train
-----
When you have enough data, you're likely better off training from scratch instead of finetuning.

.. code-block:: python

    from torch import nn, optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms, datasets
    import pytorch_lightning as pl

    # model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # data
    dataset = datasets.MNIST('./data_folder', download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    # task
    classifier = flash.Task(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

    # train
    flash.Trainer().fit(classifier, DataLoader(train), DataLoader(val))

-----

A few Built-in Tasks
--------------------

- :doc:`Task <reference/task>`
- :doc:`ImageClassification <reference/image_classification>`
- :doc:`TextClassification <reference/text_classification>`
- :doc:`TabularClassification <reference/tabular_classification>`

-----

Contribute a task
-----------------
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases.
But we're looking for incredible contributors like you to submit new tasks!

Join our `Slack <https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A>`_ to get help becoming a contributor!
