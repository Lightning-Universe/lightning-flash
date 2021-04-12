########
Registry
########

.. _registry:

********************
Available Registries
********************

Registries are mapping a name and metadata to a function.
It helps organize code and make the functions accessible all across the ``Flash`` codebase.
Each Flash ``Task`` can have several registries as static attributes.
It enables to quickly experiment with your backbone functions or use our long list of available backbones.

Example::

    from flash.vision import ImageClassifier
    from flash.core.registry import FlashRegistry

    @MyImageClassifier.backbones(name="username/my_backbone")
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

    # The new key should be listed in available backbones
    print(MyImageClassifier.available_backbones())
    # out: ["username/my_backbone"]

    # Create a model with your backbone !
    model = MyImageClassifier(backbone="username/my_backbone")

Your custom functions can be registered within a :class:`~flash.core.registry.FlashRegistry` as a decorator or directly.

Example::

    # Create a registry
    backbones = FlashRegistry("backbones")

    # Option 1: Used with partial.
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

    # HINT 1: Use `from functools import partial` if you want to store some arguments.
    backbones(fn=fn, name="username/my_backbone")


    # Option 2: Using decorator.
    @backbones(name="username/my_backbone")
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

Flash provides already populated registries containing lot of available backbones.

Example::

    from flash.vision.backbones import IMAGE_CLASSIFIER_BACKBONES, OBJ_DETECTION_BACKBONES

    print(IMAGE_CLASSIFIER_BACKBONES.available_models())
    """ out:
    ['adv_inception_v3', 'cspdarknet53', 'cspdarknet53_iabn', 430+.., 'xception71']
    """




**************
Flash Registry
**************


FlashRegistry
_____________

.. autoclass:: flash.core.registry.FlashRegistry
   :members:
