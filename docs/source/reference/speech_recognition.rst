.. customcarditem::
   :header: Speech Recognition
   :card_description: Learn to recognize speech Flash (speech-to-text) and train a model on the TIMIT corpus.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/speech_recognition.svg
   :tags: Audio,Speech-Recognition,NLP

.. _speech_recognition:

##################
Speech Recognition
##################

********
The Task
********

Speech recognition is the task of classifying audio into a text transcription. We rely on `Wav2Vec <https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/>`_ as our backbone, fine-tuned on labeled transcriptions for speech to text.
Wav2Vec is pre-trained on thousand of hours of unlabeled audio, providing a strong baseline when fine-tuning to downstream tasks such as Speech Recognition.

-----

*******
Example
*******

Let's fine-tune the model onto our own labeled audio transcription data:

Here's the structure our CSV file:

.. code-block::

    file,text
    "/path/to/file_1.wav","what was said in file 1."
    "/path/to/file_2.wav","what was said in file 2."
    "/path/to/file_3.wav","what was said in file 3."
    ...

Alternatively, here is the structure of our JSON file:

.. code-block::

    {"file": "/path/to/file_1.wav", "text": "what was said in file 1."}
    {"file": "/path/to/file_2.wav", "text": "what was said in file 2."}
    {"file": "/path/to/file_3.wav", "text": "what was said in file 3."}

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData`.
We select a pre-trained Wav2Vec backbone to use for our :class:`~flash.audio.speech_recognition.model.SpeechRecognition` and finetune on a subset of the `TIMIT corpus <https://catalog.ldc.upenn.edu/LDC93S1>`__.
The backbone can be any Wav2Vec model from `HuggingFace transformers <https://huggingface.co/models?search=wav2vec>`__.
Next, we use the trained :class:`~flash.audio.speech_recognition.model.SpeechRecognition` for inference and save the model.
Here's the full example:

.. literalinclude:: ../../../examples/speech_recognition.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The speech recognition task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash speech_recognition

To view configuration options and options for running the speech recognition task with your own data, use:

.. code-block:: bash

    flash speech_recognition --help

------

*******
Serving
*******

The :class:`~flash.audio.speech_recognition.model.SpeechRecognition` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../examples/serve/speech_recognition/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../examples/serve/speech_recognition/client.py
    :language: python
    :lines: 14-
