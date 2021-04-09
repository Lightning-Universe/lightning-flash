########
Registry
########

.. _registry:

********************
Available Registries
********************

Registries are mapping from a name and metadata to a function.
It helps organize code and make the functions accessible all across the ``Flash`` codebase.

Example::

    from flash.vision.backbones import IMAGE_CLASSIFIER_BACKBONES, OBJ_DETECTION_BACKBONES

    @IMAGE_CLASSIFIER_BACKBONES(name="username/my_backbone"):
    def fn(args_1, ... args_n):
        return backbone, num_features

    _fn = IMAGE_CLASSIFIER_BACKBONES.get("username/my_backbone")
    backbone, num_features = _fn(args_1, ..., args_n)


Each Flash ``Task`` can have several registries as static attributes.

Example::

    from flash.vision import ImageClassifier
    from flash.core.registry import FlashRegistry

    class MyImageClassifier(ImageClassifier):

        # set the registry as a static attribute
        backbones = FlashRegistry("backbones")

    # Option 1: Used with partial.
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

    # HINT 1: Use `from functools import partial` if you want to store some arguments.
    MyImageClassifier.backbones(fn=fn, name="username/my_backbone")


    # Option 2: Using decorator.
    @MyImageClassifier.backbones(name="username/my_backbone")
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

    # The new key should be listed in available backbones
    print(MyImageClassifier.available_backbones())
    # out: ["username/my_backbone"]

    # Create a model with your backbone !
    model = MyImageClassifier(backbone="username/my_backbone")

**************
Flash Registry
**************


FlashRegistry
_____________

.. autoclass:: flash.core.registry.FlashRegistry
   :members:
