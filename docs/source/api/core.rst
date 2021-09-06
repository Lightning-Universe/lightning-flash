##########
flash.core
##########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

flash.core.adapter
__________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.adapter.Adapter
    ~flash.core.adapter.AdapterTask

flash.core.classification
_________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.classification.Classes
    ~flash.core.classification.ClassificationSerializer
    ~flash.core.classification.ClassificationTask
    ~flash.core.classification.FiftyOneLabels
    ~flash.core.classification.Labels
    ~flash.core.classification.Logits
    ~flash.core.classification.PredsClassificationSerializer
    ~flash.core.classification.Probabilities

flash.core.finetuning
_____________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.finetuning.FlashBaseFinetuning
    ~flash.core.finetuning.FreezeUnfreeze
    ~flash.core.finetuning.NoFreeze
    ~flash.core.finetuning.UnfreezeMilestones

flash.core.integrations.fiftyone
________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.integrations.fiftyone.utils.visualize

flash.core.integrations.icevision
_________________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.integrations.icevision.transforms.IceVisionTransformAdapter
    ~flash.core.integrations.icevision.transforms.default_transforms
    ~flash.core.integrations.icevision.transforms.train_default_transforms

flash.core.model
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.model.BenchmarkConvergenceCI
    ~flash.core.model.CheckDependenciesMeta
    ~flash.core.model.ModuleWrapperBase
    ~flash.core.model.DatasetProcessor
    ~flash.core.model.Task

flash.core.registry
___________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.registry.FlashRegistry
    ~flash.core.registry.ExternalRegistry
    ~flash.core.registry.ConcatRegistry

flash.core.optimizers
_____________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.optimizers.LARS
    ~flash.core.optimizers.LAMB
    ~flash.core.optimizers.LinearWarmupCosineAnnealingLR

Utilities
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.trainer.from_argparse_args
    ~flash.core.utilities.apply_func.get_callable_name
    ~flash.core.utilities.apply_func.get_callable_dict
    ~flash.core.model.predict_context
