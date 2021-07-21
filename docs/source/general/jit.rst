#######################
TorchScript JIT Support
#######################

.. _jit:

We test all of our tasks for compatibility with :mod:`torch.jit`.
This table gives a breakdown of the supported features.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Task
     - :func:`torch.jit.script`
     - :func:`torch.jit.trace`
     - :func:`torch.jit.save`
   * - :class:`~flash.image.classification.model.ImageClassifier`
     - Yes
     - Yes
     - Yes
   * - :class:`~flash.image.detection.model.ObjectDetector`
     - Yes
     - No
     - Yes
   * - :class:`~flash.image.embedding.model.ImageEmbedder`
     - Yes
     - Yes
     - Yes
   * - :class:`~flash.image.segmentation.model.SemanticSegmentation`
     - No
     - Yes
     - Yes
   * - :class:`~flash.image.style_transfer.model.StyleTransfer`
     - No
     - Yes
     - Yes
   * - :class:`~flash.tabular.classification.model.TabularClassifier`
     - No
     - Yes
     - No
   * - :class:`~flash.text.classification.model.TextClassifier`
     - No
     - Yes :sup:`*`
     - Yes
   * - :class:`~flash.text.seq2seq.summarization.model.SummarizationTask`
     - No
     - Yes
     - Yes
   * - :class:`~flash.text.seq2seq.translation.model.TranslationTask`
     - No
     - Yes
     - Yes
   * - :class:`~flash.video.classification.model.VideoClassifier`
     - No
     - Yes
     - Yes

:sup:`*` with ``strict=False``
