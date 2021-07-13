##########
flash.core
##########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

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

flash.core.integration.fiftyone
_______________________________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.integrations.fiftyone.utils.visualize

flash.core.model
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.model.BenchmarkConvergenceCI
    ~flash.core.model.CheckDependenciesMeta
    ~flash.core.model.Task

flash.core.registry
___________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.core.registry.FlashRegistry

Utilities
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~flash.core.trainer.from_argparse_args
    ~flash.core.utilities.apply_func.get_callable_name
    ~flash.core.utilities.apply_func.get_callable_dict
    ~flash.core.model.predict_context
