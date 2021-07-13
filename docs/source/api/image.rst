###########
flash.image
###########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

.. currentmodule:: flash.image

Classification
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~classification.model.ImageClassifier
    ~classification.data.ImageClassificationData
    ~classification.data.ImageClassificationPreprocess

    classification.data.MatplotlibVisualization

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template:

    classification.transforms.default_transforms
    classification.transforms.train_default_transforms

Detection
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~detection.model.ObjectDetector
    ~detection.data.ObjectDetectionData

    detection.data.COCODataSource
    detection.data.ObjectDetectionFiftyOneDataSource
    detection.data.ObjectDetectionPreprocess
    detection.finetuning.ObjectDetectionFineTuning
    detection.model.ObjectDetector
    detection.serialization.DetectionLabels
    detection.serialization.FiftyOneDetectionLabels

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template:

    detection.transforms.collate
    detection.transforms.default_transforms

Embedding
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~embedding.model.ImageEmbedder

Segmentation
____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~segmentation.model.SemanticSegmentation
    ~segmentation.data.SemanticSegmentationData
    ~segmentation.data.SemanticSegmentationPreprocess

    segmentation.data.SegmentationMatplotlibVisualization
    segmentation.data.SemanticSegmentationNumpyDataSource
    segmentation.data.SemanticSegmentationTensorDataSource
    segmentation.data.SemanticSegmentationPathsDataSource
    segmentation.data.SemanticSegmentationFiftyOneDataSource
    segmentation.data.SemanticSegmentationDeserializer
    segmentation.model.SemanticSegmentationPostprocess
    segmentation.serialization.FiftyOneSegmentationLabels
    segmentation.serialization.SegmentationLabels

.. autosummary::
    :toctree: generated/
    :nosignatures:

    segmentation.transforms.default_transforms
    segmentation.transforms.prepare_target
    segmentation.transforms.train_default_transforms

Style Transfer
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~style_transfer.model.StyleTransfer
    ~style_transfer.data.StyleTransferData
    ~style_transfer.data.StyleTransferPreprocess

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~style_transfer.utils.raise_not_supported

flash.image.data
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~data.ImageDeserializer
    ~data.ImageFiftyOneDataSource
    ~data.ImageNumpyDataSource
    ~data.ImagePathsDataSource
    ~data.ImageTensorDataSource

flash.image.backbones
_____________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~backbones.catch_url_error
    ~backbones.dino_deits16
    ~backbones.dino_deits8
    ~backbones.dino_vitb16
    ~backbones.dino_vitb8
