####
Data
####

.. _data:

.. image:: https://miro.medium.com/max/1050/1*f_oNA5pSbtOO4AD8EFuTXg.gif
  :width: 600
  :alt: DataFlow Gif


***********
Terminology
***********

Here are common terms you need to be familiar with:

.. list-table:: Terminology
   :widths: 20 80
   :header-rows: 1

   * - Term
     - Definition
   * - :class:`~flash.core.data.process.Deserializer`
     - The :class:`~flash.core.data.process.Deserializer` provides a single :meth:`~flash.core.data.process.Deserializer.deserialize` method.
   * - :class:`~flash.core.data.data_module.DataModule`
     - The :class:`~flash.core.data.data_module.DataModule` contains the datasets, transforms and dataloaders.
   * - :class:`~flash.core.data.data_pipeline.DataPipeline`
     - The :class:`~flash.core.data.data_pipeline.DataPipeline` is Flash internal object to manage :class:`~flash.core.data.Deserializer`, :class:`~flash.core.data.data_source.DataSource`, :class:`~flash.core.data.io.input_transform.InputTransform`, :class:`~flash.core.data.io.output_transform.OutputTransform`, and :class:`~flash.core.data.io.output.Output` objects.
   * - :class:`~flash.core.data.data_source.DataSource`
     - The :class:`~flash.core.data.data_source.DataSource` provides :meth:`~flash.core.data.data_source.DataSource.load_data` and :meth:`~flash.core.data.data_source.DataSource.load_sample` hooks for creating data sets from metadata (such as folder names).
   * - :class:`~flash.core.data.io.input_transform.InputTransform`
     - The :class:`~flash.core.data.io.input_transform.InputTransform` provides a simple hook-based API to encapsulate your pre-processing logic.
        These hooks (such as :meth:`~flash.core.data.io.input_transform.InputTransform.pre_tensor_transform`) enable transformations to be applied to your data at every point along the pipeline (including on the device).
        The :class:`~flash.core.data.data_pipeline.DataPipeline` contains a system to call the right hooks when needed.
        The :class:`~flash.core.data.io.input_transform.InputTransform` hooks can be either overridden directly or provided as a dictionary of transforms (mapping hook name to callable transform).
   * - :class:`~flash.core.data.io.output_transform.OutputTransform`
     - The :class:`~flash.core.data.io.output_transform.OutputTransform` provides a simple hook-based API to encapsulate your post-processing logic.
        The :class:`~flash.core.data.io.output_transform.OutputTransform` hooks cover from model outputs to predictions export.
   * - :class:`~flash.core.data.io.output.Output`
     - The :class:`~flash.core.data.io.output.Output` provides a single :meth:`~flash.core.data.io.output.Output.serialize` method that is used to convert model outputs (after the :class:`~flash.core.data.io.output_transform.OutputTransform`) to the desired output format during prediction.


*******************************************
How to use out-of-the-box Flash DataModules
*******************************************

Flash provides several DataModules with helpers functions.
Check out the :ref:`image_classification` section (or the sections for any of our other tasks) to learn more.

***************
Data Processing
***************

Currently, it is common practice to implement a :class:`torch.utils.data.Dataset`
and provide it to a :class:`torch.utils.data.DataLoader`.
However, after model training, it requires a lot of engineering overhead to make inference on raw data and deploy the model in production environment.
Usually, extra processing logic should be added to bridge the gap between training data and raw data.

The :class:`~flash.core.data.data_source.DataSource` class can be used to generate data sets from multiple sources (e.g. folders, numpy, etc.), that can then all be transformed in the same way.

The :class:`~flash.core.data.io.input_transform.InputTransform` and :class:`~flash.core.data.io.output_transform.OutputTransform` classes can be used to manage the input and output transforms.
The :class:`~flash.core.data.io.output.Output` class provides the logic for converting :class:`~flash.core.data.io.output_transform.OutputTransform` outputs to the desired predict format (e.g. classes, labels, probabilities, etc.).

By providing a series of hooks that can be overridden with custom data processing logic (or just targeted with transforms),
Flash gives the user much more granular control over their data processing flow.

Here are the primary advantages:

*  Making inference on raw data simple
*  Make the code more readable, modular and self-contained
*  Data Augmentation experimentation is simpler


To change the processing behavior only on specific stages for a given hook,
you can prefix each of the :class:`~flash.core.data.io.input_transform.InputTransform` and  :class:`~flash.core.data.io.output_transform.OutputTransform`
hooks by adding ``train``, ``val``, ``test`` or ``predict``.

Check out :class:`~flash.core.data.io.input_transform.InputTransform` for some examples.

*************************************
How to customize existing DataModules
*************************************

Any Flash :class:`~flash.core.data.data_module.DataModule` can be created directly from datasets using the :meth:`~flash.core.data.data_module.DataModule.from_datasets` like this:

.. code-block:: python

    from flash import DataModule, Trainer

    data_module = DataModule.from_datasets(train_dataset=MyDataset())
    trainer = Trainer()
    trainer.fit(model, data_module=data_module)


The :class:`~flash.core.data.data_module.DataModule` provides additional ``classmethod`` helpers (``from_*``) for loading data from various sources.
In each ``from_*`` method, the :class:`~flash.core.data.data_module.DataModule` internally retrieves the correct :class:`~flash.core.data.data_source.DataSource` to use from the :class:`~flash.core.data.io.input_transform.InputTransform`.
Flash :class:`~flash.core.data.auto_dataset.AutoDataset` instances are created from the :class:`~flash.core.data.data_source.DataSource` for train, val, test, and predict.
The :class:`~flash.core.data.data_module.DataModule` populates the ``DataLoader`` for each stage with the corresponding :class:`~flash.core.data.auto_dataset.AutoDataset`.

**************************************
Customize preprocessing of DataModules
**************************************

The :class:`~flash.core.data.io.input_transform.InputTransform` contains the processing logic related to a given task.
Each :class:`~flash.core.data.io.input_transform.InputTransform` provides some default transforms through the :meth:`~flash.core.data.io.input_transform.InputTransform.default_transforms` method.
Users can easily override these by providing their own transforms to the :class:`~flash.core.data.data_module.DataModule`.
Here's an example:

.. code-block:: python

    from flash.core.data.transforms import ApplyToKeys
    from flash.image import ImageClassificationData, ImageClassifier

    transform = {"to_tensor_transform": ApplyToKeys("input", my_to_tensor_transform)}

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    )

Alternatively, the user may directly override the hooks for their needs like this:

.. code-block:: python

    from typing import Any, Dict
    from flash.image import ImageClassificationData, ImageClassifier, ImageClassificationInputTransform


    class CustomImageClassificationInputTransform(ImageClassificationInputTransform):
        def to_tensor_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample["input"] = my_to_tensor_transform(sample["input"])
            return sample


    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
        input_transform=CustomImageClassificationInputTransform(),
    )


*********************************************
Create your own InputTransform and DataModule
*********************************************

The example below shows a very simple ``ImageClassificationInputTransform`` with a single ``ImageClassificationFoldersDataSource`` and an ``ImageClassificationDataModule``.

1. User-Facing API design
_________________________

Designing an easy-to-use API is key. This is the first and most important step.
We want the ``ImageClassificationDataModule`` to generate a dataset from folders of images arranged in this way.

Example::

    train/dog/xxx.png
    train/dog/xxy.png
    train/dog/xxz.png
    train/cat/123.png
    train/cat/nsdf3.png
    train/cat/asd932.png

Example::

    dm = ImageClassificationDataModule.from_folders(
        train_folder="./data/train",
        val_folder="./data/val",
        test_folder="./data/test",
        predict_folder="./data/predict",
    )

    model = ImageClassifier(...)
    trainer = Trainer(...)

    trainer.fit(model, dm)

2. The DataSource
_________________

We start by implementing the ``ImageClassificationFoldersDataSource``.
The ``load_data`` method will produce a list of files and targets from the given directory.
The ``load_sample`` method will load the given file as a ``PIL.Image``.
Here's the full ``ImageClassificationFoldersDataSource``:

.. code-block:: python

    from PIL import Image
    from torchvision.datasets.folder import make_dataset
    from typing import Any, Dict
    from flash.core.data.data_source import DataSource, DefaultDataKeys


    class ImageClassificationFoldersDataSource(DataSource):
        def load_data(self, folder: str, dataset: Any) -> Iterable:
            # The dataset is optional but can be useful to save some metadata.

            # `metadata` contains the image path and its corresponding label
            # with the following structure:
            # [(image_path_1, label_1), ... (image_path_n, label_n)].
            metadata = make_dataset(folder)

            # for the train `AutoDataset`, we want to store the `num_classes`.
            if self.training:
                dataset.num_classes = len(np.unique([m[1] for m in metadata]))

            return [
                {
                    DefaultDataKeys.INPUT: file,
                    DefaultDataKeys.TARGET: target,
                }
                for file, target in metadata
            ]

        def predict_load_data(self, predict_folder: str) -> Iterable:
            # This returns [image_path_1, ... image_path_m].
            return [{DefaultDataKeys.INPUT: file} for file in os.listdir(folder)]

        def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
            sample[DefaultDataKeys.INPUT] = Image.open(sample[DefaultDataKeys.INPUT])
            return sample

.. note:: We return samples as dictionaries using the :class:`~flash.core.data.data_source.DefaultDataKeys` by convention. This is the recommended (although not required) way to represent data in Flash.

3. The InputTransform
_____________________

Next, implement your custom ``ImageClassificationInputTransform`` with some default transforms and a reference to the data source:

.. code-block:: python

    from typing import Any, Callable, Dict, Optional
    from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources
    from flash.core.data.io.input_transform import InputTransform
    import torchvision.transforms.functional as T

    # Subclass `InputTransform`
    class ImageClassificationInputTransform(InputTransform):
        def __init__(
            self,
            train_transform: Optional[Dict[str, Callable]] = None,
            val_transform: Optional[Dict[str, Callable]] = None,
            test_transform: Optional[Dict[str, Callable]] = None,
            predict_transform: Optional[Dict[str, Callable]] = None,
        ):
            super().__init__(
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                predict_transform=predict_transform,
                data_sources={
                    DefaultDataSources.FOLDERS: ImageClassificationFoldersDataSource(),
                },
                default_data_source=DefaultDataSources.FOLDERS,
            )

        def get_state_dict(self) -> Dict[str, Any]:
            return {**self.transforms}

        @classmethod
        def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
            return cls(**state_dict)

        def default_transforms(self) -> Dict[str, Callable]:
            return {"to_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.to_tensor)}

4. The DataModule
_________________

Finally, let's implement the ``ImageClassificationDataModule``.
We get the ``from_folders`` classmethod for free as we've registered a ``DefaultDataSources.FOLDERS`` data source in our ``ImageClassificationInputTransform``.
All we need to do is attach our :class:`~flash.core.data.io.input_transform.InputTransform` class like this:

.. code-block:: python

    from flash import DataModule


    class ImageClassificationDataModule(DataModule):

        # Set `input_transform_cls` with your custom `InputTransform`.
        input_transform_cls = ImageClassificationInputTransform


******************************
How it works behind the scenes
******************************

DataSource
__________

.. note::
    The :meth:`~flash.core.data.data_source.DataSource.load_data` and
    :meth:`~flash.core.data.data_source.DataSource.load_sample` will be used to generate an
    :class:`~flash.core.data.auto_dataset.AutoDataset` object.

Here is the :class:`~flash.core.data.auto_dataset.AutoDataset` pseudo-code.

.. code-block:: python

    class AutoDataset:
        def __init__(
            self,
            data: List[Any],  # output of `DataSource.load_data`
            data_source: DataSource,
            running_stage: RunningStage,
        ):

            self.data = data
            self.data_source = data_source

        def __getitem__(self, index: int):
            return self.data_source.load_sample(self.data[index])

        def __len__(self):
            return len(self.data)

InputTransform
______________

.. note::

    The :meth:`~flash.core.data.io.input_transform.InputTransform.pre_tensor_transform`,
    :meth:`~flash.core.data.io.input_transform.InputTransform.to_tensor_transform`,
    :meth:`~flash.core.data.io.input_transform.InputTransform.post_tensor_transform`,
    :meth:`~flash.core.data.io.input_transform.InputTransform.collate`,
    :meth:`~flash.core.data.io.input_transform.InputTransform.per_batch_transform` are injected as the
    :paramref:`torch.utils.data.DataLoader.collate_fn` function of the DataLoader.

Here is the pseudo code using the input transform hooks name.
Flash takes care of calling the right hooks for each stage.

Example::

    # This will be wrapped into a :class:`~flash.core.data.io.input_transform.flash.core.data.io.input_transform._InputTransformProcessor`.
    def collate_fn(samples: Sequence[Any]) -> Any:

        # This will be wrapped into a :class:`~flash.core.data.io.input_transform._InputTransformSequential`
        for sample in samples:
            sample = pre_tensor_transform(sample)
            sample = to_tensor_transform(sample)
            sample = post_tensor_transform(sample)

        samples = type(samples)(samples)

        # if :func:`flash.core.data.io.input_transform.InputTransform.per_sample_transform_on_device` hook is overridden,
        # those functions below will be no-ops

        samples = collate(samples)
        samples = per_batch_transform(samples)
        return samples

    dataloader = DataLoader(dataset, collate_fn=collate_fn)

.. note::

    The ``per_sample_transform_on_device``, ``collate``, ``per_batch_transform_on_device`` are injected
    after the ``LightningModule`` ``transfer_batch_to_device`` hook.

Here is the pseudo code using the input transform hooks name.
Flash takes care of calling the right hooks for each stage.

Example::

    # This will be wrapped into a :class:`~flash.core.data.io.input_transform._InputTransformProcessor`
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


OutputTransform and Output
__________________________


Once the predictions have been generated by the Flash :class:`~flash.core.model.Task`, the Flash
:class:`~flash.core.data.data_pipeline.DataPipeline` will execute the :class:`~flash.core.data.io.output_transform.OutputTransform` hooks and the
:class:`~flash.core.data.io.output.Output` behind the scenes.

First, the :meth:`~flash.core.data.io.output_transform.OutputTransform.per_batch_transform` hooks will be applied on the batch predictions.
Then, the :meth:`~flash.core.data.io.output_transform.OutputTransform.uncollate` will split the batch into individual predictions.
Next, the :meth:`~flash.core.data.io.output_transform.OutputTransform.per_sample_transform` will be applied on each prediction.
Finally, the :meth:`~flash.core.data.io.output.Output.transform` method will be called to serialize the predictions.

.. note:: The transform can be applied either on device or ``CPU``.

Here is the pseudo-code:

Example::

    # This will be wrapped into a :class:`~flash.core.data.batch._OutputTransformProcessor`
    def uncollate_fn(batch: Any) -> Any:

        batch = per_batch_transform(batch)

        samples = uncollate(batch)

        samples = [per_sample_transform(sample) for sample in samples]

        return [output.transform(sample) for sample in samples]

    predictions = lightning_module(data)
    return uncollate_fn(predictions)
