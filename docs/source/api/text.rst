##########
flash.text
##########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

Classification
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.text.classification.model.TextClassifier
    ~flash.text.classification.data.TextClassificationData

    flash.text.classification.data.TextClassificationPostprocess
    flash.text.classification.data.TextClassificationPreprocess
    flash.text.classification.data.TextCSVDataSource
    flash.text.classification.data.TextDataSource
    flash.text.classification.data.TextDeserializer
    flash.text.classification.data.TextFileDataSource
    flash.text.classification.data.TextJSONDataSource
    flash.text.classification.data.TextSentencesDataSource

Question Answering
__________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.text.seq2seq.question_answering.model.QuestionAnsweringTask
    ~flash.text.seq2seq.question_answering.data.QuestionAnsweringData

    flash.text.seq2seq.question_answering.data.QuestionAnsweringPreprocess

Summarization
_____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.text.seq2seq.summarization.model.SummarizationTask
    ~flash.text.seq2seq.summarization.data.SummarizationData

    flash.text.seq2seq.summarization.data.SummarizationPreprocess

Translation
___________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.text.seq2seq.translation.model.TranslationTask
    ~flash.text.seq2seq.translation.data.TranslationData

    flash.text.seq2seq.translation.data.TranslationPreprocess

General Seq2Seq
_______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~flash.text.seq2seq.core.model.Seq2SeqTask
    ~flash.text.seq2seq.core.data.Seq2SeqData
    ~flash.text.seq2seq.core.finetuning.Seq2SeqFreezeEmbeddings

    flash.text.seq2seq.core.data.Seq2SeqBackboneState
    flash.text.seq2seq.core.data.Seq2SeqCSVDataSource
    flash.text.seq2seq.core.data.Seq2SeqDataSource
    flash.text.seq2seq.core.data.Seq2SeqFileDataSource
    flash.text.seq2seq.core.data.Seq2SeqJSONDataSource
    flash.text.seq2seq.core.data.Seq2SeqPostprocess
    flash.text.seq2seq.core.data.Seq2SeqPreprocess
    flash.text.seq2seq.core.data.Seq2SeqSentencesDataSource
    flash.text.seq2seq.core.metrics.BLEUScore
    flash.text.seq2seq.core.metrics.RougeBatchAggregator
    flash.text.seq2seq.core.metrics.RougeMetric
