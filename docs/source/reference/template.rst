
.. _template:

########
Template
########

********
The Task
********

Here you should add a description of your task. For example:
Classification is the task of assigning one of a number of classes to each data point.

------

*******
Example
*******

.. note::

    Here you should add a short intro to your example, and then use ``literalinclude`` to add it.
    To make it simple, you can fill in this template.

Let's look at the task of <describe the task> using the <data set used in the example>.
The dataset contains <describe the data>.
Here's an outline:

.. code-block::

    <present the folder structure of the data or some data samples here>

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the <link to the DataModule with ``:class:``>.
We select a pre-trained backbone to use for our <link to the Task with ``:class:``> and finetune on the <name of the data set> data.
We then use the trained <link to the Task with ``:class:``> for inference.
Finally, we save the model.
Here's the full example:

<include the example with ``literalinclude``>

.. literalinclude:: ../../../examples/template.py
    :language: python
    :lines: 14-
