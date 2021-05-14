.. _contributing_optional:

***************
Optional Extras
***************

transforms.py
=============

Sometimes you'd like to have quite a few transforms by default (standard augmentations, normalization, etc.).
If you do then, for better organization, you can define a `transforms.py` which houses your default transforms to be referenced in your `Preprocess`.
Take a look at `vision/classification/transforms.py` for an example.

backbones.py
============

In Flash, we love to provide as much access to the state-of-the-art as we can.
To this end, we've created the `FlashRegistry` and the backbones API.
These allow you to register backbones for your task that can be selected by the user.
The backbones can come from anywhere as long as you can register a function that loads the backbone.
If you want to configure some backbones for your task, it's best practice to include these in a `backbones.py` file.
Take a look at `vision/backbones.py` for an example, and have a look at `vision/classification/model.py` to see how these can be added to your `Task`.

serialization.py
================

Sometimes you want to give the user some control over their prediction format.
`Postprocess` can do the heavy lifting (anything you always want to apply to the predictions), but one or more custom `Serializer` implementations can be used to convert the predictions to a desired output format.
A good example is in classification; sometimes we'd like the classes, sometimes the logits, sometimes the labels, you get the idea.
You should add your `Serializer` implementations in a `serialization.py` file and set a good default in your `Task`.
