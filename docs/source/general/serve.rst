.. _serve:

###########
Flash Serve
###########

Flash Serve is a library to easily serve models in production.

***********
Terminology
***********

Here are common terms you need to be familiar with:

.. list-table:: Terminology
   :widths: 20 80
   :header-rows: 1

   * - Term
     - Definition
   * - de-serialization
     - Transform data encoded as text into tensors
   * - inference function
     - A function taking the decoded tensors and forward them through the model to produce predictions.
   * - serialization
     - Transform the predictions tensors back to a text encoding.
   * - :class:`~flash.core.serve.ModelComponent`
     - The :class:`~flash.core.serve.ModelComponent` contains the de-serialization, inference and serialization functions.
   * - :class:`~flash.core.serve.Servable`
     - The :class:`~flash.core.serve.Servable` is an helper track the asset file related to a model
   * - :class:`~flash.core.serve.Composition`
     - The :class:`~flash.core.serve.Composition` defines the computations / endpoints to create & run
   * - :func:`~flash.core.serve.decorators.expose`
     - The :func:`~flash.core.serve.decorators.expose` function is a python decorator used to
       augment the :class:`~flash.core.serve.ModelComponent` inference function with de-serialization, serialization.


*******
Example
*******

In this tutorial, we will serve a Resnet18  from the `PyTorchVision library <https://github.com/pytorch/vision>`_ in 3 steps.

The entire tutorial can be found under ``examples/serve/generic``.

Introduction
============


Traditionally, an inference pipeline is made out of 3 steps:

* ``de-serialization``: Transform data encoded as text into tensors.
* ``inference function``: A function taking the decoded tensors and forward them through the model to produce predictions.
* ``serialization``: Transform the predictions tensors back as text.

In this example, we will implement only the inference function as Flash Serve already provides some built-in ``de-serialization`` and ``serialization`` functions with :class:`~flash.core.serve.types.image.Image`


Step 1 - Create a ModelComponent
================================

Inside ``inference_serve.py``,
we will implement a ``ClassificationInference`` class, which overrides :class:`~flash.core.serve.ModelComponent`.

First, we need make the following imports:

.. code-block::

    import torch
    import torchvision

    from flash.core.serve import Composition, Servable, ModelComponent, expose
    from flash.core.serve.types import Image, Label


.. image:: https://pl-flash-data.s3.amazonaws.com/assets/serve/data_serving_flow.png
  :width: 100%
  :alt: Data Serving Flow


To implement ``ClassificationInference``, we need to implement a method responsible for ``inference function`` and decorated with the :func:`~flash.core.serve.decorators.expose` function.

The name of the inference method isn't constrained, but we will use ``classify`` as appropriate in this example.

Our classify function will take a tensor image, apply some normalization on it, and forward it through the model.

.. code-block::

    def classify(img):
        img = img.float() / 255
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
        img = (img - mean) / std
        img = img.permute(0, 3, 2, 1)
        out = self.model(img)
        return out.argmax()


The :func:`~flash.core.serve.decorators.expose` is a python decorator extending the decorated function with the ``de-serialization``, ``serialization`` steps.

.. note:: Flash Serve was designed this way to enable several models to be chained together by removing the decorator.

The :func:`~flash.core.serve.decorators.expose` function takes 2 arguments:

* ``inputs``: Dictionary mapping the decorated function inputs to :class:`~flash.core.serve.types.base.BaseType` objects.
* ``outputs``: Dictionary mapping the decorated function outputs to :class:`~flash.core.serve.types.base.BaseType` objects.

A :class:`~flash.core.serve.types.base.BaseType` is a python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_
which implements a ``serialize`` and ``deserialize`` function.

.. note:: Flash Serve has already several :class:`~flash.core.serve.types.base.BaseType` built-in such as :class:`~flash.core.serve.types.image.Image` or :class:`~flash.core.serve.types.text.Text`.

.. code-block::


    class ClassificationInference(ModelComponent):
        def __init__(self, model: Servable):
            self.model = model

        @expose(
            inputs={"img": Image()},
            outputs={"prediction": Label(path="imagenet_labels.txt")},
        )
        def classify(self, img):
            img = img.float() / 255
            mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
            std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
            img = (img - mean) / std
            img = img.permute(0, 3, 2, 1)
            out = self.model(img)
            return out.argmax()


Step 2 - Create a scripted Model
================================

Using the `PyTorchVision library <https://github.com/pytorch/vision>`_, we create a ``resnet18`` and use torch.jit.script to script the model.


.. note:: TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

.. code-block::

    model = torchvision.models.resnet18(pretrained=True).eval()
    torch.jit.script(model).save("resnet.pt")

Step 3 - Serve the model
========================

The :class:`~flash.core.serve.Servable` takes as argument the path to the TorchScripted model and then will be passed to our ``ClassificationInference`` class.

The ``ClassificationInference`` instance will be passed as argument to a :class:`~flash.core.serve.Composition` class.

Once the :class:`~flash.core.serve.Composition` class is instantiated, just call its :func:`~flash.core.serve.Composition.serve` method.

.. code-block::

    resnet = Servable("resnet.pt")
    comp = ClassificationInference(resnet)
    composition = Composition(classification=comp)
    composition.serve()


Launching the server.
=====================

In Terminal 1
^^^^^^^^^^^^^^

Just run:

.. code-block::

    python inference_server.py

And you should see this in your terminal

.. image:: https://pl-flash-data.s3.amazonaws.com/assets/serve/inference_server.png
  :width: 100%
  :alt: Data Serving Flow


You should also see an Swagger UI already built for you at ``http://127.0.0.1:8000/docs``

.. image:: https://pl-flash-data.s3.amazonaws.com/assets/serve/swagger_ui.png
  :width: 100%
  :alt: Data Serving Flow


In Terminal 2
^^^^^^^^^^^^^^

Run this script from another terminal:

.. code-block::

    import base64
    from pathlib import Path

    import requests

    with Path("fish.jpg").open("rb") as f:
        imgstr = base64.b64encode(f.read()).decode("UTF-8")

    body = {"session": "UUID", "payload": {"img": {"data": imgstr}}}
    resp = requests.post("http://127.0.0.1:8000/predict", json=body)
    print(resp.json())
    # {'session': 'UUID', 'result': {'prediction': 'goldfish, Carassius auratus'}}


Credits to @rlizzo, @hhsecond, @lantiga, @luiscape for building Flash Serve Engine.
