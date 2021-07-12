###########
flash.image
###########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

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

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template:

    flash.image.classification.transforms.default_transforms
    flash.image.classification.transforms.train_default_transforms

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

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template:

    flash.image.detection.transforms.collate
    flash.image.detection.transforms.default_transforms

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

.. autosummary::
    :toctree: generated/
    :nosignatures:

    flash.image.segmentation.transforms.default_transforms
    flash.image.segmentation.transforms.prepare_target
    flash.image.segmentation.transforms.train_default_transforms

Style Transfer
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.style_transfer.model.StyleTransfer
    ~flash.image.style_transfer.data.StyleTransferData
    ~flash.image.style_transfer.data.StyleTransferPreprocess

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.image.style_transfer.utils.raise_not_supported

flash.image.data
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.image.data.ImageDeserializer
    ~flash.image.data.ImageFiftyOneDataSource
    ~flash.image.data.ImageNumpyDataSource
    ~flash.image.data.ImagePathsDataSource
    ~flash.image.data.ImageTensorDataSource

flash.image.backbones
_____________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.image.backbones.catch_url_error
    ~flash.image.backbones.dino_deits16
    ~flash.image.backbones.dino_deits8
    ~flash.image.backbones.dino_vitb16
    ~flash.image.backbones.dino_vitb8
