.. _contributing_task:

********
The Task
********

Once you've implemented a Flash :class:`~flash.core.data.data_module.DataModule` and some backbones, you should implement your :class:`~flash.core.model.Task` in `model.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/template/classification/model.py>`_.
The :class:`~flash.core.model.Task` is responsible for: setting up the backbone, performing the forward pass of the model, and calculating the loss and any metrics.
Remember that, under the hood, the Flash :class:`~flash.core.model.Task` is simply a :any:`pytorch_lightning:lightning_module` with some helpful defaults.

To build your task, you can start by overriding the base :class:`~flash.core.model.Task` or any of the existing :class:`~flash.core.model.Task` implementations.
For example, in our scikit-learn example, we can just override :class:`~flash.core.classification.ClassificationTask` which provides good defaults for classification.

You should attach your backbones registry as a class attribute like this:

.. code-block:: python

    class TemplateSKLearnClassifier(ClassificationTask):
        backbones: FlashRegistry = TEMPLATE_BACKBONES

Model architecture and hyper-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the :meth:`~flash.core.model.Task.__init__`, you will need to configure defaults for the:

- loss function
- optimizer
- metrics
- backbone / model

You will also need to create the backbone from the registry and create the model head.
Here's the code:

.. literalinclude:: ../../../flash/template/classification/model.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassifier.__init__

.. note:: We call :meth:`~pytorch_lightning.core.lightning.LightningModule.save_hyperparameters` to log the arguments to the ``__init__`` as hyperparameters. Read more `here <https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html#lightningmodule-hyperparameters>`_.

Adding the model routines
^^^^^^^^^^^^^^^^^^^^^^^^^

You should override the ``{train,val,test,predict}_step`` methods.
The default ``{train,val,test,predict}_step`` implementations in :class:`~flash.core.model.Task` expect a tuple containing the input (to be passed to the model) and target (to be used when computing the loss), and should be suitable for most applications.
In our template example, we just extract the input and target from the input mapping and forward them to the ``super`` methods.
Here's the code for the ``training_step``:

.. literalinclude:: ../../../flash/template/classification/model.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassifier.training_step

We use the same code for the ``validation_step`` and ``test_step``.
For ``predict_step`` we don't need the targets, so our code looks like this:

.. literalinclude:: ../../../flash/template/classification/model.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassifier.predict_step

.. note:: You can completely replace the ``{train,val,test,predict}_step`` methods (that is, without a call to ``super``) if you need more custom behaviour for your :class:`~flash.core.model.Task` at a particular stage.

Finally, we use our backbone and head in a custom forward pass:

.. literalinclude:: ../../../flash/template/classification/model.py
    :language: python
    :dedent: 4
    :pyobject: TemplateSKLearnClassifier.forward

------

Now that you've got your task, take a look at some :ref:`optional advanced features you can add <contributing_optional>` or go ahead and :ref:`create some examples showing your task in action! <contributing_examples>`
