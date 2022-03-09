.. _formatting_classification_targets:

*********************************
Formatting Classification Targets
*********************************

This guide details the different target formats supported by classification tasks in Flash.
By default, the target format and any additional metadata (``labels``, ``num_classes``, ``multi_label``) will be inferred from your training data.
You can override this behaviour by passing your own :class:`~flash.core.data.utilities.classification.TargetFormatter` using the ``target_formatter`` argument.

.. testsetup:: targets

    import numpy as np
    from PIL import Image

    rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
    _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]

Single Label
############

Classification targets are described as single label (``DataModule.multi_label = False``) if each data sample corresponds to a single class.

Class Indexes
_____________

Targets formatted as class indexes are represented by a single number, e.g. ``train_targets = [0, 1, 0]``.
No ``labels`` will be inferred.
The inferred ``num_classes`` is the maximum index plus one (we assume that class indexes are zero-based).
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
    >>> datamodule.labels is None
    True
    >>> datamodule.multi_label
    False

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.SingleNumericTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import SingleNumericTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[0, 1, 0],
    ...     target_formatter=SingleNumericTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    False

Labels
______

Targets formatted as labels are represented by a single string, e.g. ``train_targets = ["cat", "dog", "cat"]``.
The inferred ``labels`` will be the unique labels in the train targets sorted alphanumerically.
The inferred ``num_classes`` is the number of labels.
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

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.SingleLabelTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import SingleLabelTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "dog", "cat"],
    ...     target_formatter=SingleLabelTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    False

One-hot Binaries
________________

Targets formatted as one-hot binaries are represented by a binary list with a single index (the target class index) set to ``1``, e.g. ``train_targets = [[1, 0], [0, 1], [1, 0]]``.
No ``labels`` will be inferred.
The inferred ``num_classes`` is the length of the binary list.
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
    >>> datamodule.labels is None
    True
    >>> datamodule.multi_label
    False

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.SingleBinaryTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import SingleBinaryTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[1, 0], [0, 1], [1, 0]],
    ...     target_formatter=SingleLabelTargetFormatter(labels=["dog", "cat"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    2
    >>> datamodule.labels
    ['dog', 'cat']
    >>> datamodule.multi_label
    False

Multi Label
###########

Classification targets are described as multi label (``DataModule.multi_label = True``) if each data sample corresponds to zero or more (and perhaps many) classes.

Class Indexes
_____________

Targets formatted as multi label class indexes are represented by a list of class indexes, e.g. ``train_targets = [[0], [0, 1], [1, 2]]``.
No ``labels`` will be inferred.
The inferred ``num_classes`` is the maximum target value plus one (we assume that targets are zero-based).
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[0], [0, 1], [1, 2]],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels is None
    True
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.MultiNumericTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import MultiNumericTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[0], [0, 1], [1, 2]],
    ...     target_formatter=MultiNumericTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Labels
______

Targets formatted as multi label are represented by a list of strings, e.g. ``train_targets = [["cat"], ["cat", "dog"], ["dog", "rabbit"]]``.
The inferred ``labels`` will be the unique labels in the train targets sorted alphanumerically.
The inferred ``num_classes`` is the number of labels.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[["cat"], ["cat", "dog"], ["dog", "rabbit"]],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['cat', 'dog', 'rabbit']
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.MultiLabelTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import MultiLabelTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[["cat"], ["cat", "dog"], ["dog", "rabbit"]],
    ...     target_formatter=MultiLabelTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Comma Delimited
_______________

Targets formatted as comma delimited mutli label are given as comma delimited strings, e.g. ``train_targets = ["cat", "cat,dog", "dog,rabbit"]``.
The inferred ``labels`` will be the unique labels in the train targets sorted alphanumerically.
The inferred ``num_classes`` is the number of labels.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "cat,dog", "dog,rabbit"],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['cat', 'dog', 'rabbit']
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.CommaDelimitedMultiLabelTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import CommaDelimitedMultiLabelTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "cat,dog", "dog,rabbit"],
    ...     target_formatter=CommaDelimitedMultiLabelTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Space Delimited
_______________

Targets formatted as space delimited mutli label are given as space delimited strings, e.g. ``train_targets = ["cat", "cat dog", "dog rabbit"]``.
The inferred ``labels`` will be the unique labels in the train targets sorted alphanumerically.
The inferred ``num_classes`` is the number of labels.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "cat dog", "dog rabbit"],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['cat', 'dog', 'rabbit']
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.SpaceDelimitedTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import SpaceDelimitedTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=["cat", "cat dog", "dog rabbit"],
    ...     target_formatter=SpaceDelimitedTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Multi-hot Binaries
__________________

Targets formatted as one-hot binaries are represented by a binary list with a zero or more indices (the target class indices) set to ``1``, e.g. ``train_targets = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]``.
No ``labels`` will be inferred.
The inferred ``num_classes`` is the length of the binary list.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels is None
    True
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.MultiBinaryTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> from flash.core.data.utilities.classification import MultiBinaryTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    ...     target_formatter=MultiBinaryTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Multi-label Soft Targets
________________________

Multi-label soft targets are represented by a list of floats, e.g. ``train_targets = [[0.1, 0, 0], [0.9, 0.7, 0], [0, 0.5, 0.6]]``.
No ``labels`` will be inferred.
The inferred ``num_classes`` is the length of the list.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassificationData
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[0.1, 0, 0], [0.9, 0.7, 0], [0, 0.5, 0.6]],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels is None
    True
    >>> datamodule.multi_label
    True

Alternatively, you can provide a :class:`~flash.core.data.utilities.classification.MultiSoftTargetFormatter` to override the behaviour.
Here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassificationData
    >>> from flash.core.data.utilities.classification import MultiSoftTargetFormatter
    >>> datamodule = ImageClassificationData.from_files(
    ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
    ...     train_targets=[[0.1, 0, 0], [0.9, 0.7, 0], [0, 0.5, 0.6]],
    ...     target_formatter=MultiSoftTargetFormatter(labels=["dog", "cat", "rabbit"]),
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True

Also, you can use Pandas DataFrame, here's an example:

.. doctest:: targets

    >>> from flash import Trainer
    >>> from flash.image import ImageClassifier, ImageClassificationData
    >>> df = pd.DataFrame(
    ...     [["image_1.png", 0.1, 0, 0], ["image_2.png", 0.9, 0.7, 0], ["image_3.png"0, 0.5, 0.6]],
    ...     columns=["image", "dog", "cat", "rabbit"])
    ...
    >>> datamodule = ImageClassificationData.from_data_frame(
    ...     train_data_frame=df,
    ...     ["dog", "cat", "rabbit"],
    ...     transform_kwargs=dict(image_size=(128, 128)),
    ...     batch_size=2,
    ... )
    >>> datamodule.num_classes
    3
    >>> datamodule.labels
    ['dog', 'cat', 'rabbit']
    >>> datamodule.multi_label
    True
