####
Data
####

.. _datapipeline:

DataPipeline
------------

To make tasks work for inference, one must create a ``DataPipeline``.
The ``flash.core.data.DataPipeline`` exposes 6 hooks to override:

.. code:: python

    class DataPipeline:
        """
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
        """

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

        def uncollate(self, batch: Any) -> ny:
            """Override to convert a batch to a set of samples"""
            samples = batch
            return samples

        def after_uncollate(self, samples: Any) -> Any:
            """Override to apply transformations to samples"""
            return samplesA
