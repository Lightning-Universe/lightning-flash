.. customcarditem::
   :header: Audio Classification
   :card_description: Learn to classify audio spectrogram images with Flash and build an example classifier for the UrbanSound8k data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/audio_classification.svg
   :tags: Audio,Classification

.. _audio_classification:

####################
Audio Classification
####################

********
The Task
********

The task of identifying what is in an audio file is called audio classification.
Typically, Audio Classification is used to identify audio files containing sounds or words.
The task predicts which ‘class’ the sound or words most likely belongs to with a degree of certainty.
A class is a label that describes the sounds in an audio file, such as ‘children_playing’, ‘jackhammer’, ‘siren’ etc.

------

*******
Example
*******

Let's look at the task of predicting whether audio file contains sounds of an airconditioner, carhorn, childrenplaying, dogbark, drilling, engingeidling, gunshot, jackhammer, siren, or street_music using the UrbanSound8k spectrogram images dataset.
The dataset contains ``train``, ``val``  and ``test`` folders, and then each folder contains a **airconditioner** folder, with spectrograms generated from air-conditioner sounds, **siren** folder with spectrograms generated from siren sounds and the same goes for the other classes.

.. code-block::

    urban8k_images
    ├── train
    │   ├── air_conditioner
    │   ├── car_horn
    │   ├── children_playing
    │   ├── dog_bark
    │   ├── drilling
    │   ├── engine_idling
    │   ├── gun_shot
    │   ├── jackhammer
    │   ├── siren
    │   └── street_music
    ├── test
    │   ├── air_conditioner
    │   ├── car_horn
    │   ├── children_playing
    │   ├── dog_bark
    │   ├── drilling
    │   ├── engine_idling
    │   ├── gun_shot
    │   ├── jackhammer
    │   ├── siren
    │   └── street_music
    └── val
        ├── air_conditioner
        ├── car_horn
        ├── children_playing
        ├── dog_bark
        ├── drilling
        ├── engine_idling
        ├── gun_shot
        ├── jackhammer
        ├── siren
        └── street_music

            ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.audio.classification.data.AudioClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.image.classification.model.ImageClassifier` and fine-tune on the UrbanSound8k spectrogram images data.
We then use the trained :class:`~flash.image.classification.model.ImageClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/audio_classification.py
    :language: python
    :lines: 14-

------

**********
Flash Zero
**********

The audio classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash audio_classification

To view configuration options and options for running the audio classifier with your own data, use:

.. code-block:: bash

    flash audio_classification --help

------

************
Loading Data
************

.. autodatasources:: flash.audio.classification.data AudioClassificationData

    {% extends "base.rst" %}
    {% block from_datasets %}
    {{ super() }}

    .. note::

        The ``__getitem__`` of your datasets should return a dictionary with ``"input"`` and ``"target"`` keys which map to the input spectrogram image (as a NumPy array) and the target (as an int or list of ints) respectively.
    {% endblock %}
