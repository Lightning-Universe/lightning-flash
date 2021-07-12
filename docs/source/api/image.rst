###########
flash.image
###########

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Tasks
=====

Classification
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.classification.model.ImageClassifier
    ~flash.image.classification.data.ImageClassificationData
    ~flash.image.classification.data.ImageClassificationPreprocess

    flash.image.classification.data.MatplotlibVisualization

Detection
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.detection.model.ObjectDetector
    ~flash.image.detection.data.ObjectDetectionData

    flash.image.detection.data.COCODataSource
    flash.image.detection.data.ObjectDetectionFiftyOneDataSource
    flash.image.detection.data.ObjectDetectionPreprocess
    flash.image.detection.finetuning.ObjectDetectionFineTuning
    flash.image.detection.model.ObjectDetector
    flash.image.detection.serialization.DetectionLabels
    flash.image.detection.serialization.FiftyOneDetectionLabels

Embedding
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.embedding.model.ImageEmbedder

Segmentation
____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.segmentation.model.SemanticSegmentation
    ~flash.image.segmentation.data.SemanticSegmentationData
    ~flash.image.segmentation.data.SemanticSegmentationPreprocess

    flash.image.segmentation.data.SegmentationMatplotlibVisualization
    flash.image.segmentation.data.SemanticSegmentationNumpyDataSource
    flash.image.segmentation.data.SemanticSegmentationTensorDataSource
    flash.image.segmentation.data.SemanticSegmentationPathsDataSource
    flash.image.segmentation.data.SemanticSegmentationFiftyOneDataSource
    flash.image.segmentation.data.SemanticSegmentationDeserializer
    flash.image.segmentation.model.SemanticSegmentationPostprocess
    flash.image.segmentation.serialization.FiftyOneSegmentationLabels
    flash.image.segmentation.serialization.SegmentationLabels

Style Transfer
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.style_transfer.model.StyleTransfer
    ~flash.image.style_transfer.data.StyleTransferData
    ~flash.image.style_transfer.data.StyleTransferPreprocess

flash.image.data
================

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.data.ImageDeserializer
    ~flash.image.data.ImageFiftyOneDataSource
    ~flash.image.data.ImageNumpyDataSource
    ~flash.image.data.ImagePathsDataSource
    ~flash.image.data.ImageTensorDataSource
