.. _formatting_classification_targets:

*********************************
Formatting Classification Targets
*********************************

This guide details the different target formats supported by classification tasks in Flash.
By default, the target format and any additional metadata (`labels`, `num_classes`, `multi_label`) will be inferred from your training data.

.. testsetup:: targets

    import numpy as np
    from PIL import Image

    rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
    _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]

Single Numeric
______________

Single numeric targets are represented by a single integer (`multi_label = False`).
No `labels` will be inferred.
The inferred `num_classes` is the maximum target value plus one (we assume that targets are zero-based).
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[0, 1, 0],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    2
    >>> datamodule.labels
    None
    >>> datamodule.multi_label
    False

Single Labels
_____________

Single labels are targets represented by a single string (`multi_label = False`).
The inferred `labels` will be the unique labels in the train targets sorted alphanumerically.
The inferred `num_classes` is the number of labels.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "dog", "cat"],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    2
    >>> datamodule.labels
    ['cat', 'dog']
    >>> datamodule.multi_label
    False

Single Binary
_____________

Single binary targets are represented by a one-hot encoded binary list (`multi_label = False`).
No `labels` will be inferred.
The inferred `num_classes` is the length of the binary list.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[1, 0], [0, 1], [1, 0]],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    2
    >>> datamodule.labels
    None
    >>> datamodule.multi_label
    False
