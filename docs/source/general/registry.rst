
.. _registry:

########
Registry
########

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

.. testcode:: registries

    from functools import partial

    from flash import Task
    from flash.core.registry import FlashRegistry

2. Init a Registry
__________________

It is good practice to associate one or multiple registry to a Task as follow:

.. testcode:: registries

    # creating a custom `Task` with its own registry
    class MyImageClassifier(Task):
        backbones = FlashRegistry("backbones")

        def __init__(
            self,
            backbone: str = "resnet18",
            pretrained: bool = True,
        ):
            ...

            self.backbone, self.num_features = self.backbones.get(backbone)(pretrained=pretrained)


3. Adding new functions
_______________________

Your custom functions can be registered within a :class:`~flash.core.registry.FlashRegistry` as a decorator or directly.

.. testcode:: registries

    # Option 1: Used with partial.
    def fn(backbone: str, pretrained: bool = True):
        # Create backbone and backbone output dimension (`num_features`)
        backbone, num_features = None, None
        return backbone, num_features


    # HINT 1: Use `from functools import partial` if you want to store some arguments.
    MyImageClassifier.backbones(fn=partial(fn, backbone="my_backbone"), name="username/partial_backbone")


    # Option 2: Using decorator.
    @MyImageClassifier.backbones(name="username/decorated_backbone")
    def fn(pretrained: bool = True):
        # Create backbone and backbone output dimension (`num_features`)
        backbone, num_features = None, None
        return backbone, num_features

4. Accessing registered functions
_________________________________

You can now access your function from your task!

.. testcode:: registries

    # 3.b Optional: List available backbones
    print(MyImageClassifier.available_backbones())

    # 4. Build the model
    model = MyImageClassifier(backbone="username/decorated_backbone")

Here's the output:

.. testoutput:: registries

    ['username/decorated_backbone', 'username/partial_backbone']

5. Pre-registered backbones
___________________________

Flash provides populated registries containing lots of available backbones.

Example::

    from flash.image.backbones import OBJ_DETECTION_BACKBONES
    from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES

    print(IMAGE_CLASSIFIER_BACKBONES.available_keys())
    """ out:
    ['adv_inception_v3', 'cspdarknet53', 'cspdarknet53_iabn', 430+.., 'xception71']
    """
