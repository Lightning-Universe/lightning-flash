###############
flash.core.data
###############

.. contents::
    :depth: 1
    :local:
    :backlinks: top

flash.core.data.base_viz
________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.base_viz.BaseVisualization

flash.core.data.batch
________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.data.batch.default_uncollate

flash.core.data.callback
________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.callback.BaseDataFetcher
    ~flash.core.data.callback.ControlFlow
    ~flash.core.data.callback.FlashCallback

flash.core.data.data_module
___________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.data_module.DataModule

flash.core.data.utilities.classification
________________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.utilities.classification.TargetFormatter
    ~flash.core.data.utilities.classification.SingleNumericTargetFormatter
    ~flash.core.data.utilities.classification.SingleLabelTargetFormatter
    ~flash.core.data.utilities.classification.SingleBinaryTargetFormatter
    ~flash.core.data.utilities.classification.MultiNumericTargetFormatter
    ~flash.core.data.utilities.classification.MultiLabelTargetFormatter
    ~flash.core.data.utilities.classification.CommaDelimitedMultiLabelTargetFormatter
    ~flash.core.data.utilities.classification.SpaceDelimitedTargetFormatter
    ~flash.core.data.utilities.classification.MultiBinaryTargetFormatter
    ~flash.core.data.utilities.classification.get_target_formatter

flash.core.data.utilities.collate
_________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.data.utilities.collate.default_collate

flash.core.data.properties
__________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.properties.Properties

flash.core.data.splits
______________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.splits.SplitDataset

flash.core.data.transforms
__________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.transforms.ApplyToKeys
    ~flash.core.data.transforms.KorniaParallelTransforms

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.data.transforms.kornia_collate

flash.core.data.utils
_____________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.utils.FuncModule

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.data.utils.convert_to_modules
    ~flash.core.data.utils.download_data

flash.core.data.io.input
___________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.io.input.Input
    ~flash.core.data.io.input.DataKeys
    ~flash.core.data.io.input.InputFormat

flash.core.data.io.classification_input
_______________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.io.classification_input.ClassificationInputMixin

flash.core.data.io.input_transform
__________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.io.input_transform.InputTransform

flash.core.data.io.output
_________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.io.output.Output

flash.core.data.io.output_transform
___________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.data.io.output_transform.OutputTransform
