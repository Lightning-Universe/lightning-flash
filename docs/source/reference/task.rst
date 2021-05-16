############
General Task
############

A majority of data science problems that involve machine learning can be tackled using Task. With Task you can:

- Pass an arbitrary model
- Pass an arbitrary loss
- Pass an arbitrary optimizer

*****************************
Example: Image Classification
*****************************

.. literalinclude:: ../../../flash_examples/generic_task.py
    :language: python
    :lines: 14-

-----

*************
API reference
*************

.. _task:

Task
----

.. autoclass:: flash.core.model.Task
    :members:
    :noindex:
    :exclude-members: forward
