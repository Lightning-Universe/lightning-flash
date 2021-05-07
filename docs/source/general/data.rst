####
Data
####

.. _data:


***********
Terminology
***********

Here are common terms you need to be familiar with:

.. list-table:: Terminology
   :widths: 20 80
   :header-rows: 1

   * - Term
     - Definition
   * - :class:`~flash.data.data_module.DataModule`
     - The :class:`~flash.data.data_module.DataModule` contains the dataset, transforms and dataloaders.
   * - :class:`~flash.data.data_pipeline.DataPipeline`
     - The :class:`~flash.data.data_pipeline.DataPipeline` is Flash internal object to manage :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` objects.
   * - :class:`~flash.data.data_source.DataSource`
     - The :class:`~flash.data.data_source.DataSource` provides a hook-based API for creating data sets.
   * - :class:`~flash.data.process.Preprocess`
     - The :class:`~flash.data.process.Preprocess` provides a simple hook-based API to encapsulate your pre-processing logic.
        The :class:`~flash.data.process.Preprocess` provides multiple hooks such as :meth:`~flash.data.process.Preprocess.load_data`
        and :meth:`~flash.data.process.Preprocess.load_sample` which are used to replace a traditional `Dataset` logic.
        Flash DataPipeline contains a system to call the right hooks when needed.
        The :class:`~flash.data.process.Preprocess` hooks covers from data-loading to model forwarding.
   * - :class:`~flash.data.process.Postprocess`
     - The :class:`~flash.data.process.Postprocess` provides a simple hook-based API to encapsulate your post-processing logic.
        The :class:`~flash.data.process.Postprocess` hooks cover from model outputs to predictions export.
   * - :class:`~flash.data.process.Serializer`
     - The :class:`~flash.data.process.Serializer` provides a single ``serialize`` method that is used to convert model outputs (after the :class:`~flash.data.process.Postprocess`) to the desired output format during prediction.

*******************************************
How to use out-of-the-box flashdatamodules
*******************************************

Flash provides several DataModules with helpers functions.
Checkout the :ref:`image_classification` section or any other tasks to learn more about them.

***************
Data Processing
***************

Currently, it is common practice to implement a `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
and provide them to a `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

However, after model training, it requires a lot of engineering overhead to make inference on raw data and deploy the model in production environnement.
Usually, extra processing logic should be added to bridge the gap between training data and raw data.

The :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` classes can be used to
store the data as well as the preprocessing and postprocessing transforms. The :class:`~flash.data.process.Serializer`
class provides the logic for converting :class:`~flash.data.process.Postprocess` outputs to the desired predict format
(e.g. classes, labels, probabilites, etc.).

By providing a series of hooks that can be overridden with custom data processing logic,
the user has much more granular control over their data processing flow.

Here are the primary advantages:

*  Making inference on raw data simple
*  Make the code more readable, modular and self-contained
*  Data Augmentation experimentation is simpler


To change the processing behavior only on specific stages for a given hook,
you can prefix each of the :class:`~flash.data.process.Preprocess` and  :class:`~flash.data.process.Postprocess`
hooks by adding ``train``, ``val``, ``test`` or ``predict``.

Check out :class:`~flash.data.process.Preprocess` for some examples.

.. note::

    ``[WIP]`` We are currently working on a new feature to make :class:`~flash.data.process.Preprocess`

    and :class:`~flash.data.process.Postprocess` automatically deployable from checkpoints as

    ``Endpoints`` or ``BatchTransformJob``. Stay tuned !

*************************************
How to customize existing datamodules
*************************************

Flash DataModule can receive directly dataset as follow:

Example::

    from flash.data.data_module import DataModule

    dm = DataModule(train_dataset=MyDataset(train=True))
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, data_module=dm)

In order to customize Flash to your need, you need to know what are :class:`~flash.data.data_module.DataModule`
and :class:`~flash.data.process.Preprocess` responsibilities.

.. note::

    At this point, we strongly encourage the readers to quickly check the :class:`~flash.data.process.Preprocess` API before getting further.

The :class:`~flash.data.data_module.DataModule` provides ``classmethod`` helpers to build
:class:`~flash.data.process.Preprocess` and :class:`~flash.data.data_pipeline.DataPipeline`,
generate Flash Internal :class:`~flash.data.auto_dataset.AutoDataset` and populate DataLoaders with them.

The :class:`~flash.data.process.Preprocess` contains the processing logic related to a given task. Users can easily override hooks
to customize a built-in :class:`~flash.data.process.Preprocess` for their needs.

Example::

    from flash.vision import ImageClassificationData, ImageClassifier, ImageClassificationPreprocess

    class CustomImageClassificationPreprocess(ImageClassificationPreprocess):

        # Assuming you have images in numpy format,
        # just override ``load_sample`` hook and add your own logic.
        @staticmethod
        def load_sample(sample) -> Tuple[Image.Image, int]:
            # By default, ``ImageClassificationPreprocess`` expects
            # ``.png`` or ``.jpg`` to be loaded into PIL Image.
            numpy_image_path, label = sample
            return np.load(numpy_image_path), sample

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
        preprocess=CustomImageClassificationPreprocess(),
    )


******************************
Custom Preprocess + Datamodule
******************************

The example below shows a very simple ``ImageClassificationPreprocess`` with a ``ImageClassificationDataModule``.

1. User-Facing API design
_________________________

Designing an easy to use API is key. This is the first and most important step.

We want the ``ImageClassificationDataModule`` to generate a dataset from folders of images arranged in this way.

Example::

    train/dog/xxx.png
    train/dog/xxy.png
    train/dog/xxz.png
    train/cat/123.png
    train/cat/nsdf3.png
    train/cat/asd932.png

Example::

    preprocess = ...

    dm = ImageClassificationDataModule.from_folders(
        train_folder="./data/train",
        val_folder="./data/val",
        test_folder="./data/test",
        predict_folder="./data/predict",
        preprocess=preprocess,
    )

    model = ImageClassifier(...)
    trainer = Trainer(...)

    trainer.fit(model, dm)

2. The DataModule
__________________

Secondly, let's implement the ``ImageClassificationDataModule`` from_folders classmethod.

Example::

    from flash.data.data_module import DataModule

    class ImageClassificationDataModule(DataModule):

        # Set ``preprocess_cls`` with your custom ``preprocess``.
        preprocess_cls = ImageClassificationPreprocess

        @classmethod
        def from_folders(
            cls,
            train_folder: Optional[str],
            val_folder: Optional[str],
            test_folder: Optional[str],
            predict_folder: Optional[str],
            preprocess: Optional[Preprocess] = None,
            **kwargs
        ):

            # Set a custom ``Preprocess`` if none was provided
            preprocess = preprocess or cls.preprocess_cls()

            # {stage}_load_data_input will be given to your
            # ``Preprocess`` ``{stage}_load_data`` function.
            return cls.from_load_data_inputs(
                    train_load_data_input=train_folder,
                    val_load_data_input=val_folder,
                    test_load_data_input=test_folder,
                    predict_load_data_input=predict_folder,
                    preprocess=preprocess,  # DON'T FORGET TO PASS THE CREATED PREPROCESS
                    **kwargs,
            )


3. The Preprocess
__________________

Finally, implement your custom ``ImageClassificationPreprocess``.

Example::

    import os
    import numpy as np
    from flash.data.process import Preprocess
    from PIL import Image
    import torchvision.transforms as T
    from torch import Tensor
    from torchvision.datasets.folder import make_dataset

    # Subclass ``Preprocess``
    class ImageClassificationPreprocess(Preprocess):

        to_tensor = T.ToTensor()

        def load_data(self, folder: str, dataset: AutoDataset) -> Iterable:
            # The AutoDataset is optional but can be useful to save some metadata.

            # metadata contains the image path and its corresponding label with the following structure:
            # [(image_path_1, label_1), ... (image_path_n, label_n)].
            metadata = make_dataset(folder)

            # for the train ``AutoDataset``, we want to store the ``num_classes``.
            if self.training:
                dataset.num_classes = len(np.unique([m[1] for m in metadata]))

            return metadata

        def predict_load_data(self, predict_folder: str) -> Iterable:
            # This returns [image_path_1, ... image_path_m].
            return os.listdir(folder)

        def load_sample(self, sample: Union[str, Tuple[str, int]]) -> Tuple[Image, int]
            if self.predicting:
                return Image.open(image_path)
            else:
                image_path, label = sample
                return Image.open(image_path), label

        def to_tensor_transform(
            self,
            sample: Union[Image, Tuple[Image, int]]
        ) -> Union[Tensor, Tuple[Tensor, int]]:

            if self.predicting:
                return self.to_tensor(sample)
            else:
                return self.to_tensor(sample[0]), sample[1]


.. note::

    Currently, Flash Tasks are implemented using :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess`.
    However, it is not a hard requirement and one can still use :class:`~torch.data.utils.Dataset`, but we highly recommend
    using :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` instead.


*************
API reference
*************

.. _data_source:

DataSource
__________

.. autoclass:: flash.data.data_source.DataSource
    :members:


----------

.. _preprocess:

Preprocess
__________

.. autoclass:: flash.data.process.Preprocess
    :members:


----------

.. _postprocess:

Postprocess
___________


.. autoclass:: flash.data.process.Postprocess
    :members:


----------

.. _serializer:

Serializer
___________


.. autoclass:: flash.data.process.Serializer
    :members:


----------

.. _datapipeline:

DataPipeline
____________

.. autoclass:: flash.data.data_pipeline.DataPipeline
    :members:

----------

DataModule
__________

.. autoclass:: flash.data.data_module.DataModule
    :members:
        train_dataset,
        val_dataset,
        test_dataset,
        predict_dataset,
        configure_data_fetcher,
        show_train_batch,
        show_val_batch,
        show_test_batch,
        show_predict_batch,
    :exclude-members:
        autogenerate_dataset,


******************************
How it works behind the scenes
******************************

Preprocess
__________

.. note:: The ``load_data`` and ``load_sample`` will be used to generate an AutoDataset object.

Here is the ``AutoDataset`` pseudo-code.

Example::

    from pytorch_lightning.trainer.states import RunningStage

    class AutoDataset
        def __init__(
            self,
            data: Any,
            load_data: Optional[Callable] = None,
            load_sample: Optional[Callable] = None,
            data_pipeline: Optional['DataPipeline'] = None,
            running_stage: Optional[RunningStage] = None
        ) -> None:

            self.preprocess = data_pipeline._preprocess_pipeline
            self.preprocessed_data: Iterable = self.preprocess.load_data(data)

        def __getitem__(self, index):
            return self.preprocess.load_sample(self.preprocessed_data[index])

        def __len__(self):
            return len(self.preprocessed_data)

.. note::

    The ``pre_tensor_transform``, ``to_tensor_transform``, ``post_tensor_transform``, ``collate``,
    ``per_batch_transform`` are injected as the ``collate_fn`` function of the DataLoader.

Here is the pseudo code using the preprocess hooks name.
Flash takes care of calling the right hooks for each stage.

Example::

    # This will be wrapped into a :class:`~flash.data.batch._PreProcessor`.
    def collate_fn(samples: Sequence[Any]) -> Any:

        # This will be wrapped into a :class:`~flash.data.batch._Sequential`
        for sample in samples:
            sample = pre_tensor_transform(sample)
            sample = to_tensor_transform(sample)
            sample = post_tensor_transform(sample)

        samples = type(samples)(samples)

        # if :func:`flash.data.process.Preprocess.per_sample_transform_on_device` hook is overridden,
        # those functions below will be no-ops

        samples = collate(samples)
        samples = per_batch_transform(samples)
        return samples

    dataloader = DataLoader(dataset, collate_fn=collate_fn)

.. note::

    The ``per_sample_transform_on_device``, ``collate``, ``per_batch_transform_on_device`` are injected
    after the ``LightningModule`` ``transfer_batch_to_device`` hook.

Here is the pseudo code using the preprocess hooks name.
Flash takes care of calling the right hooks for each stage.

Example::

    # This will be wrapped into a :class:`~flash.data.batch._PreProcessor`
    def collate_fn(samples: Sequence[Any]) -> Any:

        # if ``per_batch_transform`` hook is overridden, those functions below will be no-ops
        samples = [per_sample_transform_on_device(sample) for sample in samples]
        samples = type(samples)(samples)
        samples = collate(samples)

        samples = per_batch_transform_on_device(samples)
        return samples

    # move the data to device
    data = lightning_module.transfer_data_to_device(data)
    data = collate_fn(data)
    predictions = lightning_module(data)


Postprocess and Serializer
__________________________


Once the predictions have been generated by the Flash :class:`~flash.core.model.Task`, the Flash
:class:`~flash.data.data_pipeline.DataPipeline` will execute the :class:`~flash.data.process.Postprocess` hooks and the
:class:`~flash.data.process.Serializer` behind the scenes.

First, the :meth:`~flash.data.process.Postprocess.per_batch_transform` hooks will be applied on the batch predictions.
Then, the :meth:`~flash.data.process.Postprocess.uncollate` will split the batch into individual predictions.
Next, the :meth:`~flash.data.process.Postprocess.per_sample_transform` will be applied on each prediction.
Finally, the :meth:`~flash.data.process.Serializer.serialize` method will be called to serialize the predictions.

.. note:: The transform can be applied either on device or ``CPU``.

Here is the pseudo-code:

Example::

    # This will be wrapped into a :class:`~flash.data.batch._PreProcessor`
    def uncollate_fn(batch: Any) -> Any:

        batch = per_batch_transform(batch)

        samples = uncollate(batch)

        samples = [per_sample_transform(sample) for sample in samples]
        # only if serializers are enabled.
        return [serialize(sample) for sample in samples]

    predictions = lightning_module(data)
    return uncollate_fn(predictions)
