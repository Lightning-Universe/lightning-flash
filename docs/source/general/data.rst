####
Data
####

.. _datapipeline:

DataPipeline
------------

To make tasks work for inference, one must create a ``Preprocess`` and ``PostProcess``.
The ``flash.data.process.Preprocess`` exposes 9 hooks to override which can specifialzed for each stage using
``train``, ``val``, ``test``, ``predict`` prefixes:

.. code:: python

    from flash.data.process import Postprocess, Preprocess
    from flash.data.data_module import DataModule
    import torchvision.transforms as T

    class ImageClassificationPreprocess(Preprocess):

        def __init__(self, to_tensor_transform, train_per_sample_transform_on_device):
            super().__init__()
            self._to_tensor = to_tensor_transform
            self._train_per_sample_transform_on_device = train_per_sample_transform_on_device

        def load_data(self, folder: str):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path: str) -> Image.Image:
            # from a file path, load the associated image
            img8Bit = np.uint8(np.random.uniform(0, 1, (64, 64, 3)) * 255.0)
            return Image.fromarray(img8Bit)

        def per_sample_to_tensor_transform(self, pil_image: Image.Image) -> torch.Tensor:
            # convert pil image into a tensor
            return self._to_tensor(pil_image)

        def train_per_sample_transform_on_device(self, sample: Any) -> Any:
            # apply an augmentation per sample on gpu for train only
            return self._train_per_sample_transform_on_device(sample)

    class CustomModel(Task):

        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

        def training_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

        def validation_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

        def test_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 64, 64])

    class CustomDataModule(DataModule):

        preprocess_cls = ImageClassificationPreprocess

        @property
        def preprocess(self):
            return self.preprocess_cls(self.to_tensor_transform, self.train_per_sample_transform_on_device)

        @classmethod
        def from_folders(
            cls, train_folder: Optional[str], val_folder: Optional[str], test_folder: Optional[str],
            predict_folder: Optional[str], to_tensor_transform: torch.nn.Module,
            train_per_sample_transform_on_device: torch.nn.Module, batch_size: int
        ):

            # attach the arguments for the preprocess onto the cls
            cls.to_tensor_transform = to_tensor_transform
            cls.train_per_sample_transform_on_device = train_per_sample_transform_on_device

            # call ``from_load_data_inputs``
            return cls.from_load_data_inputs(
                train_load_data_input=train_folder,
                valid_load_data_input=val_folder,
                test_load_data_input=test_folder,
                predict_load_data_input=predict_folder,
                batch_size=batch_size
            )

    datamodule = CustomDataModule.from_folders(
        "train_folder", "val_folder", "test_folder", None, T.ToTensor(), T.RandomHorizontalFlip(), batch_size=2
    )

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        num_sanity_val_steps=1
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)
