.. _speech_recognition:

##################
Speech Recognition
##################

********
The Task
********

Speech recognition is the task of classifying audio into a text transcription. We rely on `Wav2Vec <https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/>`_ as our backbone, fine-tuned on labeled transcriptions for speech to text.

-----

*******
Example
*******

Let's fine-tune the model onto our own labeled audio transcription data:

Here's the structure our CSV file:

.. code-block::

    file,text
    "/path/to/file_1.wav ... ","what was said in file 1."
    "/path/to/file_2.wav ... ","what was said in file 2."
    "/path/to/file_3.wav ... ","what was said in file 3."
    ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData`.
We select a pre-trained Wav2Vec backbone to use for our :class:`~flash.audio.speech_recognition.model.SpeechRecognition` and finetune on a subset of the `TIMIT corpus <https://catalog.ldc.upenn.edu/LDC93S1>`__.
The backbone can be any Wav2Vec model from `HuggingFace transformers <https://huggingface.co/models?search=wav2vec>`__.
Next, we use the trained :class:`~flash.audio.speech_recognition.model.SpeechRecognition` for inference and save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/speech_recognition.py
    :language: python
    :lines: 14-

------

*******
Serving
*******

The :class:`~flash.audio.speech_recognition.model.SpeechRecognition` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/speech_recognition/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/speech_recognition/client.py
    :language: python
    :lines: 14-
