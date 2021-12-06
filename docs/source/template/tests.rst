.. _contributing_tests:

*********
The Tests
*********

Our next step is to create some tests for our :class:`~flash.core.model.Task`.
For the ``TemplateSKLearnClassifier``, we will just create some basic tests.
You should expand on these to include tests for any specific functionality you have in your :class:`~flash.core.model.Task`.

Smoke tests
===========

We use smoke tests, usually called ``test_smoke``, throughout.
These just instantiate the class we are testing, to see that they can be created without raising any errors.

tests/examples/test_scripts.py
==============================

Before we write our custom tests, we should add out examples to the CI.
To do this, add a line for each example (``finetuning`` and ``predict``) to the annotation of ``test_example`` in `tests/examples/test_scripts.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/tests/examples/test_scripts.py>`_.
Here's how those lines look for our ``template.py`` examples:

.. code-block:: python

    pytest.param(
        "finetuning", "template.py", marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
    ),
    ...
    pytest.param(
        "predict", "template.py", marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
    ),

test_data.py
============

The most important tests in `test_data.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/tests/template/classification/test_data.py>`_ check that the ``from_*`` methods work correctly.
In the class ``TestTemplateData``, we have two of these: ``test_from_numpy`` and ``test_from_sklearn``.
In general, there should be one ``test_from_*`` method for each :class:`~flash.core.data.io.input` you have configured.

Here's the code for ``test_from_numpy``:

.. literalinclude:: ../../../tests/template/classification/test_data.py
    :language: python
    :pyobject: TestTemplateData.test_from_numpy

test_model.py
=============

In `test_model.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/tests/template/classification/test_model.py>`_, we first have ``test_forward`` and ``test_train``.
These test that tensors can be passed to the forward and that the :class:`~flash.core.model.Task` can be trained.
Here's the code for ``test_forward`` and ``test_train``:

.. literalinclude:: ../../../tests/template/classification/test_model.py
    :language: python
    :pyobject: test_forward

.. literalinclude:: ../../../tests/template/classification/test_model.py
    :language: python
    :pyobject: test_train

|

We also include tests for validating and testing: ``test_val``, and ``test_test``.
These tests are very similar to ``test_train``, but here they are for completeness:

.. literalinclude:: ../../../tests/template/classification/test_model.py
    :language: python
    :pyobject: test_val

.. literalinclude:: ../../../tests/template/classification/test_model.py
    :language: python
    :pyobject: test_test

|

We also include tests for prediction named ``test_predict_*`` for each of our data sources.
In our case, we have ``test_predict_numpy`` and ``test_predict_sklearn``.
These tests should load the data with a :class:`~flash.core.data.data_module.DataModule` and generate predictions with :func:`Trainer.predict <flash.core.trainer.Trainer.predict>`.
Here's ``test_predict_sklearn`` as an example:

.. literalinclude:: ../../../tests/template/classification/test_model.py
    :language: python
    :pyobject: test_predict_sklearn

------

Now that you've written the tests, it's time to :ref:`add some docs! <contributing_docs>`
