.. _contributing_task:

********
The Task
********

Inside ``model.py`` you just need to implement your :class:`~flash.core.model.Task`.
The :class:`~flash.core.model.Task` is responsible for the forward pass of the model.
It's just a :any:`pytorch_lightning:lightning_module` with some helpful defaults, so anything you can do inside a :any:`pytorch_lightning:lightning_module` you can do inside a :class:`~flash.core.model.Task`.

Task
^^^^

You should configure a default loss function and optimizer and some default metrics and models in your :class:`~flash.core.model.Task`.
For our scikit-learn example, we can just override :class:`~flash.core.classification.ClassificationTask` which provides these defaults for us.
You should also override the ``*_step`` methods to unpack your sample.
The default ``*_step`` implementations in :class:`~flash.core.model.Task` expect a tuple containing the input and target, and should be suitable for most applications.
In our template example, we just extract the input and target from the input mapping and forward them to the super methods.

Here's our ``TemplateSKLearnClassifier``:

.. autoclass:: flash.template.classification.model.TemplateSKLearnClassifier
    :members:

.. raw:: html

    <details>
    <summary>Source</summary>

.. literalinclude:: ../../../flash/template/classification/model.py
    :language: python
    :pyobject: TemplateSKLearnClassifier

.. raw:: html

    </details>

------

Now that you've got your task, take a look at some :ref:`optional advanced features you can add <contributing_optional>` or go ahead and :ref:`create some examples showing your task in action! <contributing_examples>`
