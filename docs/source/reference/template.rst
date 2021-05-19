
.. _template:

########
Template
########

********
The task
********

Here you should add a description of your task. For example:
Classification is the task of assigning one of a number of classes to each data point.
The :class:`~flash.template.TemplateSKLearnClassifier` is a :class:`~flash.core.model.Task` for classifying the datasets included with scikit-learn.

------

*********
Inference
*********

Here, you should add a short intro to your predict example, and then use ``literalinclude`` to add it.

.. note:: We skip the first 14 lines as they are just the copyright notice.

Our predict example uses a model pre-trained on the Iris data.

.. literalinclude:: ../../../flash_examples/predict/template.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

********
Training
********

In this section, we briefly describe the data, and then ``literalinclude`` our finetuning example.

Now we'll train on Fisher's classic Iris data.
It contains 150 records with four features (sepal length, sepal width, petal length, and petal width) in three classes (species of Iris: setosa, virginica and versicolor).

Now all we need is to train our task!

.. literalinclude:: ../../../flash_examples/finetuning/template.py
    :language: python
    :lines: 14-

------

*************
API reference
*************

We usually include the API reference for the :class:`~flash.core.model.Task` and :class:`~flash.core.data.data_module.DataModule`.
You can optionally add the other classes you've implemented.
To add the API reference, use the ``autoclass`` directive.

.. _template_classifier:

TemplateSKLearnClassifier
-------------------------

.. autoclass:: flash.template.TemplateSKLearnClassifier
    :members:
    :exclude-members: forward

.. _template_data:

TemplateData
------------

.. autoclass:: flash.template.TemplateData
