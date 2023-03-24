.. _contributing_examples:

***********
The Example
***********

Now you've implemented your task, it's time to add an example showing how cool it is!
We usually provide one example in `examples/ <https://github.com/PyTorchLightning/lightning-flash/blob/master/examples/>`_.
You can base these off of our ``template.py`` examples.

The example should:

#. download the data (we'll add the example to our CI later on, so choose a dataset small enough that it runs in reasonable time)
#. load the data into a :class:`~flash.core.data.data_module.DataModule`
#. create an instance of the :class:`~flash.core.model.Task`
#. create a :class:`~flash.core.trainer.Trainer`
#. call :meth:`~flash.core.trainer.Trainer.finetune` or :meth:`~flash.core.trainer.Trainer.fit` to train your model
#. generate predictions for a few examples
#. save the checkpoint

For our template example we don't have a pretrained backbone, so we can just call :meth:`~flash.core.trainer.Trainer.fit` rather than :meth:`~flash.core.trainer.Trainer.finetune`.
Here's the full example (`examples/template.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/examples/template.py>`_):

.. literalinclude:: ../../../examples/template.py
    :language: python
    :lines: 14-

We get this output:

.. code-block::

    ['setosa', 'virginica', 'versicolor']

------

Now that you've got an example showing your awesome task in action, it's time to :ref:`write some tests! <contributing_tests>`
