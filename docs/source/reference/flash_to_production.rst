########################
From Flash to Production
########################

Flash makes it simple to deploy models in production.

Use a :class:`~flash.FlashServeModel` and pass as arguments a :class:`~flash.Task` class and path or url to an associated checkpoint to serve.

Server Side
^^^^^^^^^^^

.. literalinclude:: ../../../flash_examples/serve/segmentic_segmentation/inference_server.py
    :language: python
    :lines: 14-


Client Side
^^^^^^^^^^^

.. literalinclude:: ../../../flash_examples/serve/segmentic_segmentation/client.py
    :language: python
    :lines: 14-
