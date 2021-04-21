
.. _multi_label_classification:

################################
Multi-label Image Classification
################################

********
The task
********
Multi-label classification is the task of assigning a number of labels from a fixed set to each data point, which can be in any modality. In this example, we will look at the task of trying to predict the movie genres from an image of the movie poster.

------

********
The data
********
The data we will use in this example is a subset of the awesome movie poster genre prediction data set from the paper "Movie Genre Classification based on Poster Images with Deep Neural Networks" by Wei-Ta Chu and Hung-Jui Guo, resized to 128 by 128.
Take a look at their paper (and please consider citing their paper if you use the data) here: `www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/ <https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/>`_.

------

*********
Inference
*********

The :class:`~flash.vision.ImageClassifier` is already pre-trained on `ImageNet <http://www.image-net.org/>`_, a dataset of over 14 million images.

We can use the :class:`~flash.vision.ImageClassifier` model (pretrained on our data) for inference on any string sequence using :func:`~flash.vision.ImageClassifier.predict`.
We can also add a simple visualisation by extending :class:`~flash.data.base_viz.BaseVisualization`, like this:

.. code-block:: python

    # import our libraries
    from typing import Any

    import torchvision.transforms.functional as T
    from torchvision.utils import make_grid

    from flash import Trainer
    from flash.data.base_viz import BaseVisualization
    from flash.data.utils import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

    # 2. Define our custom visualisation and datamodule
    class CustomViz(BaseVisualization):

        def show_per_batch_transform(self, batch: Any, _):
            images = batch[0]
            image = make_grid(images, nrow=2)
            image = T.to_pil_image(image, 'RGB')
            image.show()


    # 3. Load the model from a checkpoint
    model = ImageClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/image_classification_multi_label_model.pt",
    )

    # 4a. Predict the genres of a few movie posters!
    predictions = model.predict([
        "data/movie_posters/val/tt0361500.jpg",
        "data/movie_posters/val/tt0361748.jpg",
        "data/movie_posters/val/tt0362478.jpg",
    ])
    print(predictions)

    # 4b. Or generate predictions with a whole folder!
    datamodule = ImageClassificationData.from_folders(
        predict_folder="data/movie_posters/predict/",
        data_fetcher=CustomViz(),
        preprocess=model.preprocess,
    )

    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)

    # 5. Show some data!
    datamodule.show_predict_batch()

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

Now let's look at how we can finetune a model on the movie poster data.
Once we download the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.vision.ImageClassificationData`.

.. note:: The dataset contains ``train`` and ``validation`` folders, and then each folder contains images and a ``metadata.csv`` which stores the labels.

.. code-block::

    movie_posters
    ├── train
    │   ├── metadata.csv
    │   ├── tt0084058.jpg
    │   ├── tt0084867.jpg
    │   ...
    └── val
        ├── metadata.csv
        ├── tt0200465.jpg
        ├── tt0326965.jpg
        ...


The ``metadata.csv`` files in each folder contain our labels, so we need to create a function (``load_data``) to extract the list of images and associated labels:

.. code-block:: python

    # import our libraries
    import os
    from typing import List, Tuple

    import pandas as pd
    import torch

    genres = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "N/A", "News", "Reality-TV", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western"
    ]

    def load_data(data: str, root: str = 'data/movie_posters') -> Tuple[List[str], List[torch.Tensor]]:
        metadata = pd.read_csv(os.path.join(root, data, "metadata.csv"))

        images = []
        labels = []
        for _, row in metadata.iterrows():
            images.append(os.path.join(root, data, row['Id'] + ".jpg"))
            labels.append(torch.tensor([row[genre] for genre in genres]))

        return images, labels

Our :class:`~flash.data.process.Preprocess` overrides the :meth:`~flash.data.process.Preprocess.load_data` method to create an iterable of image paths and label tensors. The :class:`~flash.vision.classification.data.ImageClassificationPreprocess` then handles loading and augmenting the images for us!
Now all we need is three lines of code to build to train our task!

.. note:: We need set `multi_label=True` in both our :class:`~flash.Task` and our :class:`~flash.data.process.Serializer` to use a binary cross entropy loss and to process outputs correctly.

.. code-block:: python

    import flash
    from flash.core.classification import Labels
    from flash.core.finetuning import FreezeUnfreeze
    from flash.data.utils import download_data
    from flash.vision import ImageClassificationData, ImageClassifier
    from flash.vision.classification.data import ImageClassificationPreprocess

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

    # 2. Load the data
    ImageClassificationPreprocess.image_size = (128, 128)

    train_filepaths, train_labels = load_data('train')
    val_filepaths, val_labels = load_data('val')
    test_filepaths, test_labels = load_data('test')

    datamodule = ImageClassificationData.from_filepaths(
        train_filepaths=train_filepaths,
        train_labels=train_labels,
        val_filepaths=val_filepaths,
        val_labels=val_labels,
        test_filepaths=test_filepaths,
        test_labels=test_labels,
        preprocess=ImageClassificationPreprocess(),
    )

    # 3. Build the model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=len(genres),
        multi_label=True,
    )

    # 4. Create the trainer.
    trainer = flash.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, strategy=FreezeUnfreeze(unfreeze_epoch=1))

    # 6a. Predict what's on a few images!

    # Serialize predictions as labels.
    model.serializer = Labels(genres, multi_label=True)

    predictions = model.predict([
        "data/movie_posters/val/tt0361500.jpg",
        "data/movie_posters/val/tt0361748.jpg",
        "data/movie_posters/val/tt0362478.jpg",
    ])

    print(predictions)

    datamodule = ImageClassificationData.from_folders(
        predict_folder="data/movie_posters/predict/",
        preprocess=model.preprocess,
    )

    # 6b. Or generate predictions with a whole folder!
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)

    # 7. Save it!
    trainer.save_checkpoint("image_classification_multi_label_model.pt")

------

For more backbone options, see :ref:`image_classification`.
