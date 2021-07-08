###############
flash.core.data
###############

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Data Loading
____________

Data Module
===========

.. automodule:: flash.core.data.data_module
    :exclude-members: DataModule

Data Sources
============

.. automodule:: flash.core.data.data_source
    :exclude-members: DataSource

Data Processing
_______________

Data Pipeline
=============

.. automodule:: flash.core.data.data_pipeline

Data Pipeline Components
========================

.. automodule:: flash.core.data.properties

.. automodule:: flash.core.data.process
    :exclude-members: Postprocess, Preprocess, Serializer

Transforms
__________

.. currentmodule:: flash.core.data.transforms

Helpers
=======

ApplyToKeys
-----------

.. autoclass:: ApplyToKeys

merge_transforms
----------------

.. autofunction:: merge_transforms

Kornia
======

KorniaParallelTransforms
------------------------

.. autoclass:: KorniaParallelTransforms

kornia_collate
--------------

.. autofunction:: kornia_collate

Callbacks and Visualizations
____________________________

.. automodule:: flash.core.data.base_viz

.. automodule:: flash.core.data.callback

Utilities
_________

.. automodule:: flash.core.data.auto_dataset

.. automodule:: flash.core.data.batch

.. automodule:: flash.core.data.splits

.. automodule:: flash.core.data.utils
