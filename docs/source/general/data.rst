####
Data
####

.. _data:

*******************************
Using DataModule + DataPipeline
*******************************

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
