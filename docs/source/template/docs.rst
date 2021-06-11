.. _contributing_docs:

*********
The Docs
*********

The final step is to add some docs.
For each :class:`~flash.core.model.Task` in Flash, we have a docs page in `docs/source/reference <https://github.com/PyTorchLightning/lightning-flash/blob/master/docs/source/reference>`_.
You should create a ``.rst`` file there with the following:

- a brief description of the task
- the predict example
- the finetuning example
- any relevant API reference

Here are the contents of `docs/source/reference/template.rst <https://github.com/PyTorchLightning/lightning-flash/blob/master/docs/source/reference/template.rst>`_ which breaks down each of these steps:

.. literalinclude:: ../reference/template.rst
    :language: rest

:ref:`Here's the rendered doc page! <template>`

------

Once the docs are done, it's finally time to open a PR and wait for some reviews!

|

Congratulations on adding your first :class:`~flash.core.model.Task` to Flash, we hope to see you again soon!
