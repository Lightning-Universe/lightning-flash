******************************
Adding a new task from scratch
******************************

One should start by subclassing :class:`~flash.core.model.Task`.

To make this new task work for training, one should have to override the forward function.

.. code-block:: python

    from flash.core.model import Task

    class ImageClassifier(Task):

        def __init__(
            self,
            num_classes,
            backbone="resnet18",
            loss_fn: Callable = F.cross_entropy,
        ):
            super().__init__(
                model=None,
                loss_fn=loss_fn,
            )

            self.save_hyperparameters()

            self.backbone, num_features = instantiate_backbone(backbone)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(num_features, num_classes),
            )

        def forward(self, x) -> Any:
            x = self.backbone(x)
            return self.head(x)

    model = ImageClassifier(2)
    model(torch.randn(1, 3, 32, 32))

To make this new task work for inference, one would have to create a DataPipeline.
A shell `flash.core.data.datapipeline.DataPipeline` exposes 6 hooks to override.

.. code-block:: python

    class DataPipeline:
        This class purpose is to facilitate the conversion of raw data to processed or batched data and back.
        Several hooks are provided for maximum flexibility.

        collate_fn:
            - before_collate
            - collate
            - after_collate

        uncollate_fn:
            - before_uncollate
            - uncollate
            - after_uncollate

        def before_collate(self, samples: Any) -> Any:
            """Override to apply transformations to samples"""
            return samples

        def collate(self, samples: Any) -> Any:
            """Override to convert a set of samples to a batch"""
            if not isinstance(samples, Tensor):
                return default_collate(samples)
            return samples

        def after_collate(self, batch: Any) -> Any:
            """Override to apply transformations to the batch"""
            return batch

        def before_uncollate(self, batch: Any) -> Any:
            """Override to apply transformations to the batch"""
            return batch

        def uncollate(self, batch: Any) -> Any:
            """Override to convert a batch to a set of samples"""
            samples = batch
            return samples

        def after_uncollate(self, samples: Any) -> Any:
            """Override to apply transformations to samples"""
            return samples


Here is the ImageClassifierDataPipeline where `before_collate`, `before_uncollate`
and `after_uncollate` are being overriden.

.. code-block:: python

    class ImageClassifierDataPipeline(DataPipeline):

        """
        The DataPipeline should be attached to the DataModule.
        It should be pickable, so it can be saved/loaded from checkpoint for inference.
        """

        def __init__(
            self,
            train_transform: Optional[Callable] = _default_train_transforms,
            valid_transform: Optional[Callable] = _default_valid_transforms,
            use_valid_transform: bool = True,
            loader: Callable = _pil_loader
        ):
            self._train_transform = train_transform
            self._valid_transform = valid_transform
            self._use_valid_transform = use_valid_transform
            self._loader = loader

        def before_collate(self, samples: Any) -> Any:
            if _contains_any_tensor(samples):
                return samples

            if isinstance(samples, str):
                samples = [samples]
            if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
                outputs = []
                for sample in samples:
                    output = self._loader(sample)
                    transform = self._valid_transform if self._use_valid_transform else self._train_transform
                    outputs.append(transform(output))
                return outputs
            raise MisconfigurationException("The samples should either be a tensor or a list of paths.")

        def before_uncollate(self, batch: Union[torch.Tensor, tuple]) -> torch.Tensor:
            # Apply softmax over predictions
            return torch.softmax(batch, -1)

        def after_uncollate(self, samples: Any) -> Any:
            # Get the most likely class for each prediction.
            return torch.argmax(samples, -1)

Finally, let's see how model.predict works internally.
As one can observe, predict will call ``data_pipeline.collate_fn``, ``model.forward``,
``data_pipeline.uncollate_fn``. For ImageClassifierDataPipeline, ``data_pipeline.collate_fn``
will be used to convert a list of image paths to tensors.

.. code-block:: python

    def predict(
        self,
        x: Any,
        batch_idx: Optional[int] = None,
        skip_collate_fn: bool = False,
        dataloader_idx: Optional[int] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Any:
        """
        Predict function for raw data or processed data

        Args:

            x: Input to predict. Can be raw data or processed data.

            batch_idx: Batch index

            dataloader_idx: Dataloader index

            skip_collate_fn: Whether to skip the collate step.
                this is required when passing data already processed
                for the model, for example, data from a dataloader

            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions

        """
        data_pipeline = data_pipeline or self.data_pipeline
        batch = x if skip_collate_fn else data_pipeline.collate_fn(x)
        batch_x, batch_y = batch if len(batch) == 2 else (batch, None)
        predictions = self.forward(batch_x)
        return data_pipeline.uncollate_fn(predictions)  # TODO: pass batch and x
