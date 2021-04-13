########
Registry
########

.. _registry:

********************
Available Registries
********************

Registries are Flash internal key-value database to store a mapping between a name and a function.

In simple words, they are just advanced dictionary storing a function from a key string.

Registries help organize code and make the functions accessible all across the ``Flash`` codebase.
Each Flash :class:`~flash.core.model.Task` can have several registries as static attributes.

Currently, Flash uses internally registries only for backbones, but more components will be added.

1. Imports
__________

Example::

    from flash.core.registry import FlashRegistry

2. Init a Registry
__________________

It is good practice to associate one or multiple registry to a Task as follow:

Example::

    from flash.vision import ImageClassifier
    from flash.core.registry import FlashRegistry

    # creating a custom ``ImageClassifier`` with its own registry
    class MyImageClassifier(ImageClassifier):

        backbones = FlashRegistry("backbones")

3. Adding new functions
_______________________

Your custom functions can be registered within a :class:`~flash.core.registry.FlashRegistry` as a decorator or directly.

Example::

    # Option 1: Used with partial.
    def fn(backbone: str):
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

    # HINT 1: Use `from functools import partial` if you want to store some arguments.
    MyImageClassifier.backbones(fn=partial(fn, backbone="my_backbone"), name="username/my_backbone")


    # Option 2: Using decorator.
    @MyImageClassifier.backbones(name="username/my_backbone")
    def fn():
        # Create backbone and backbone output dimension (`num_features`)
        return backbone, num_features

4. Accessing registered functions
_________________________________

You can now access your function from your task!

Example::

    # 3.b Optional: List available backbones
    print(MyImageClassifier.available_backbones())
    # out: ["username/my_backbone"]

    # 4. Build the model
    model = MyImageClassifier(backbone="username/my_backbone", num_classes=2)


5. Pre-registered ones
______________________

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
