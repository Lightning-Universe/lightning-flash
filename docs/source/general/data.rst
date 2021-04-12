####
Data
####

.. _datapipeline:

DataPipeline
------------

To make tasks work for inference, one must create a ``Preprocess`` and ``PostProcess``.
The ``flash.data.process.Preprocess`` exposes 9 hooks to override which can specifialzed for each stage using
``train``, ``val``, ``test``, ``predict`` prefixes.
