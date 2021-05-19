.. _contributing_task:

********
The Task
********

Once you've implemented a Flash :class:`~flash.core.data.data_module.DataModule`, you should implement your :class:`~flash.core.model.Task`.
The :class:`~flash.core.model.Task` is responsible for: setting up the backbone, performing the forward pass of the model, and calculating the loss and any metrics.
Remember that, under the hood, the Flash :class:`~flash.core.model.Task` is simply a :any:`pytorch_lightning:lightning_module` with some helpful defaults.

Task
^^^^

When creating your :class:`~flash.core.model.Task`, you will need to configure defaults for the:

- loss function
- optimizer
- metrics
- model

You can override the base :class:`~flash.core.model.Task` or any of the existing :class:`~flash.core.model.Task` implementations.
For example, in our scikit-learn example, we can just override :class:`~flash.core.classification.ClassificationTask` which provides good defaults for classification.

|

You should also override the ``{train,val,test,predict}_step`` methods.
The default ``{train,val,test,predict}_step`` implementations in :class:`~flash.core.model.Task` expect a tuple containing the input and target, and should be suitable for most applications.
In our template example, we just extract the input and target from the input mapping and forward them to the ``super`` methods.
You can completely replace the ``{train,val,test,predict}_step`` methods (that is, without a call to ``super``) if you need more custom behaviour for your :class:`~flash.core.model.Task` at a particular stage.

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
