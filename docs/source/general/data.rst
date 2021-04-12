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
   * - DataModule
     - The :class:`~flash.data.data_module.DataModule` contains the dataset, transforms and dataloaders.
   * - :class:`~flash.data.data_pipeline.DataPipeline`
     - The :class:`~flash.data.data_pipeline.DataPipeline` is Flash internal object to manage :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` objects.
   * - :class:`~flash.data.process.Preprocess`
     - The :class:`~flash.data.process.Preprocess` provides a simple hook-based API to encapsulate your pre-processing logic.
        This enables to not rely on Dataset anymore even if supported.
        Flash DataPipeline contains a system to call the right hooks when needed.
        The :class:`~flash.data.process.Preprocess` hooks covers from data-loading to model forwarding.
   * - :class:`~flash.data.process.Postprocess`
     - The :class:`~flash.data.process.Postprocess` provides a simple hook-based API to encapsulate your post-processing logic.
        The :class:`~flash.data.process.Postprocess` hooks covers from model outputs to predictions export.

*******************************************
How to use out-of-the-box flashdatamodules
*******************************************

Flash provides several DataModules with helpers functions.
Checkout the ``Task Section`` to learn more about them.

********************************
Why Preprocess and PostProcess ?
********************************

Currently, it is common pratices to implement a :class:`~torch.data.utils.Dataset` and provide them to a :class:`~torch.data.utils.DataLoader`.

However, once the model is trained, lot of engineering work is required to enable the model
to perform predictions on ``un-processed data`` (called raw data) or easily experiment when some transforms should applied during training.
But more importantly, it is hard to make a trainer model ready for production.
The :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` have been created to resolve those issues.
By providing a series of hooks which can overridden with custom data processing logic, the user have a more granular control.
But more importantly, it makes your code more readable, modular and easy to extend.
To change the processing behaviour only on specific stages for a given hook,
you can prefix all the above hooks adding ``train``, ``val``, ``test`` or ``predict``.

.. note::

    [WIP] Once the :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` are fully implemented,

    the model should deployable for ``Endpoints`` or ``BatchTransformJob`` directly from checkpoints.

*************************************
How to customize existing datamodules
*************************************

Currently, Flash Tasks are implementing using :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess`.
However, it is not a hard requirement and one can still use :class:`~torch.data.utils.Dataset`, but we highly recommend
using :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess` instead.

Example::

    from flash.data.data_module import DataModule

    dm = DataModule(train_dataset=MyDataset(train=True))
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, data_module=dm)

In order to customize Flash to your need, you need to know what are :class:`~flash.data.data_module.DataModule`
and :class:`~flash.data.process.Preprocess` responsibilities.

.. note::

    From this point, we strongly encourage the readers to quickly check the :class:`~flash.data.process.Preprocess` API before getting further.

The :class:`~flash.data.data_module.DataModule` provides ``classmethod`` helpers to build
:class:`~flash.data.process.Preprocess` and :class:`~flash.data.data_pipeline.DataPipeline`,
generate ``AutoDataset`` and populate DataLoaders from them.

The :class:`~flash.data.process.Preprocess` contains the processing logic related to a given task. One can easily override hooks
to customize a built-in :class:`~flash.data.process.Preprocess` for his need.

Example::

    from flash.vision import ImageClassificationData, ImageClassifier, ImageClassificationPreprocess

    class CustomImageClassificationPreprocess(ImageClassificationPreprocess):

        @staticmethod
        def load_sample(sample) -> Tuple[Image.Image, int]:
            # By default, ``ImageClassificationPreprocess`` expects ``.PNG`` or ``.JPB`` to be loaded into PIL Image.
            # Assuming you have numpy image, just override this hook and add your own logic.
            numpy_image_path, label = sample
            return np.load(numpy_image_path), sample

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
        preprocess_cls=CustomImageClassificationPreprocess
    )


*****************************************************
How to build a Datamodule + Preprocess for a new task
*****************************************************

In this section, we will create a very simple ``ImageClassificationPreprocess`` with a ``ImageClassificationDataModule``.

Example::

    import os
    import numpy as np
    from flash.data.data_module import DataModule
    from flash.data.process import Preprocess
    from PIL.Image import Image
    import torchvision.transforms as T
    from torch import Tensor

    # Subclass ``Preprocess``

    class ImageClassificationPreprocess(Preprocess):

        to_tensor = T.ToTensor()

        def load_data(self, folder: str, dataset: AutoDataset) -> Iterable:
            # The AutoDataset is optional but can be useful to save some metadata.

            # metadata looks like this: [(image_path_1, label_1), ... (image_path_n, label_n)].
            metadata = make_dataset_from_folder(folder)

            # for the train ``AutoDataset``, we want to store the ``num_classes``.
            if self.training:
                dataset.num_classes = len(np.unique([m[1] for m in metadata]))

            return metadata

        def predict_load_data(self, predict_folder: str) -> Iterable:
            # This returns [image_path_1, ... image_path_m].
            return os.listdir(folder)

        def load_sample(self, sample: Union[str, Tuple[str, int]]) -> Tuple[Image, int]
            if self.predicting:
                return load_pil(image_path)
            else:
                image_path, label = sample
                return load_pil(image_path), label

        def to_tensor_transform(
            self,
            sample: Union[Image, Tuple[Image, int]]
        ) -> Union[Tensor, Tuple[Tensor, int]]:

            if self.predicting:
                return self.to_tensor(sample)
            else:
                return self.to_tensor(sample[0]), sample[1]

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
            **kwargs
        ):

            preprocess = cls.preprocess_cls()

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

    dm = ImageClassificationDataModule.from_folders("./data/train", "./data/val", "./data/test", "./data/predict")

    model = ImageClassifier(...)
    trainer = Trainer(...)

    trainer.fit(model, dm)



*************
API reference
*************

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
Surely, Flash will take care of calling the right hooks for each stage.

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
Surely, Flash will take care of calling the right hooks for each stage.

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


Postprocess
___________


Example::

    # This will be wrapped into a :class:`~flash.data.batch._PreProcessor`
    def uncollate_fn(batch: Any) -> Any:

        batch = per_batch_transform(batch)

        samples = uncollate(batch)

        return [per_sample_transform(sample) for sample in samples]

    predictions = lightning_module(data)
    return uncollate_fn(predictions)
