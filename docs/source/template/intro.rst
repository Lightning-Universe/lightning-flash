.. _contributing:

*********************
Introduction / Set-up
*********************

Welcome
=======

Before you begin, we'd like to express our gratitude to you for wanting to add a task to Flash.
With Flash our aim is to create a great user experience, enabling awesome advanced applications with just a few lines of code.
We're really pleased with what we've achieved with Flash and we hope you will be too.
Now let's dive in!

Set-up
======

The Task template is designed to guide you through contributing a task to Flash.
It contains the code, tests, and examples for a task that performs classification with a multi-layer perceptron, intended for use with the classic data sets from scikit-learn.
The Flash tasks are organized in folders by data-type (image, text, video, etc.), with sub-folders for different task types (classification, regression, etc.).

Copy the files in `flash/template/classification <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/template/classification/>`_ to a new sub-directory under the relevant data-type.
If a data-type folder already exists for your task, then a task type sub-folder should be added containing the template files.
If a data-type folder doesn't exist, then you will need to add that too.
You should also copy the files from `tests/template/classification <https://github.com/PyTorchLightning/lightning-flash/blob/master/tests/template/classification/>`_ to the corresponding data-type, task type folder in ``tests``.
For example, if you were adding an image classification task, you would do:

.. code-block:: bash

    mkdir flash/image/classification
    cp flash/template/classification/* flash/image/classification/
    mkdir tests/image/classification
    cp tests/template/classification/* tests/image/classification/

Tutorials
=========

The tutorials in this section will walk you through all of the components you need to implement (or adapt from the template) for your custom task.

- :ref:`contributing_data`: our first tutorial goes over the best practices for implementing everything you need to connect data to your task
- :ref:`contributing_backbones`: the second tutorial shows you how to create an extensible backbone registry for your task
- :ref:`contributing_task`: now that we have the data and the models, in this tutorial we create our custom task
- :ref:`contributing_optional`: this tutorial covers some optional extras you can add if needed for your particular task
- :ref:`contributing_examples`: this tutorial guides you through creating some simple examples showing your task in action
- :ref:`contributing_tests`: in this tutorial, we cover best practices for writing some tests for your new task
- :ref:`contributing_docs`: in our final tutorial, we provide a template for you to create the docs page for your task
