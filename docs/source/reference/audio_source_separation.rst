
.. _Audio_Source_Separation:

##########################
Audio Source Separation
##########################

********
The Task
********

Audio Source Separation is the process of separating a mixture (e.g. a pop band recording) into isolated sounds from individual sources (e.g. just the lead vocals).
This is essential to robust speech processing in real-world acoustic environments.
This technology is among the most studied in audio signal processing today and bear a critical role in the success of hearing aids, hands-free phones, voice command and other noise-robust audio analysis systems, and music post-production software.


------

*******
Example
*******

Let's look at the task of separating a mixture into two voice sounds.
The dataset contains:
* ``{stage}/s1``: Source of the first voice
* ``{stage}/s2``: Source of the second voice
* ``{stage}/noise``: Additional noise to make the task harder.
* ``{stage}/mix_clean``: Mixture of the two voice sounds
* ``{stage}/mix_both``: Mixture of the two voice sounds with additional noise.

.. code-block::

    📦 MiniLibriMix
     ┣ 📂 metadata
     ┃ ┣ 📜 mixture_train_mix_both.csv
     ┃ ┣ 📜 mixture_train_mix_clean.csv
     ┃ ┣ 📜 mixture_val_mix_both.csv
     ┃ ┗ 📜 mixture_val_mix_clean.csv
     ┣ 📂 train
     ┃ ┣ 📂 mix_both
     ┃ ┃ ┣ 📜 100-121669-0026_718-129597-0003.wav
     ┃ ┃ ┣ 📜 1025-92820-0032_8410-278217-0015.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 mix_clean
     ┃ ┃ ┣ 📜 100-121669-0026_718-129597-0003.wav
     ┃ ┃ ┣ 📜 1025-92820-0032_8410-278217-0015.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 noise
     ┃ ┃ ┣ 📜 100-121669-0026_718-129597-0003.wav
     ┃ ┃ ┣ 📜 1025-92820-0032_8410-278217-0015.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 s1
     ┃ ┃ ┣ 📜 100-121669-0026_718-129597-0003.wav
     ┃ ┃ ┣ 📜 1025-92820-0032_8410-278217-0015.wav
     ┃ ┃ ┣ ...
     ┃ ┗ 📂 s2
     ┃ ┃ ┣ 📜 100-121669-0026_718-129597-0003.wav
     ┃ ┃ ┣ 📜 1025-92820-0032_8410-278217-0015.wav
     ┃ ┃ ┣ ...
     ┗ 📂 val
     ┃ ┣ 📂 mix_both
     ┃ ┃ ┣ 📜 1272-128104-0006_2428-83705-0022.wav
     ┃ ┃ ┣ 📜 1272-128104-0008_1919-142785-0031.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 mix_clean
     ┃ ┃ ┣ 📜 1272-128104-0006_2428-83705-0022.wav
     ┃ ┃ ┣ 📜 1272-128104-0008_1919-142785-0031.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 noise
     ┃ ┃ ┣ 📜 1272-128104-0006_2428-83705-0022.wav
     ┃ ┃ ┣ 📜 1272-128104-0008_1919-142785-0031.wav
     ┃ ┃ ┣ ...
     ┃ ┣ 📂 s1
     ┃ ┃ ┣ 📜 1272-128104-0006_2428-83705-0022.wav
     ┃ ┃ ┣ 📜 1272-128104-0008_1919-142785-0031.wav
     ┃ ┃ ┣ ...
     ┃ ┗ 📂 s2
     ┃ ┃ ┣ 📜 1272-128104-0006_2428-83705-0022.wav
     ┃ ┃ ┣ 📜 1272-128104-0008_1919-142785-0031.wav
     ┃ ┃ ┣ ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.classification.data.ImageClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.image.classification.model.ImageClassifier` and fine-tune on the hymenoptera data.
We then use the trained :class:`~flash.image.classification.model.ImageClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/audio_source_separation.py
    :language: python
    :lines: 14-
