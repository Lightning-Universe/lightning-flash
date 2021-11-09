##########
flash.text
##########

.. contents::
    :depth: 1
    :local:
    :backlinks: top

.. currentmodule:: flash.text

Classification
______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~classification.model.TextClassifier
    ~classification.data.TextClassificationData

    classification.data.TextClassificationOutputTransform
    classification.data.TextClassificationInputTransform
    classification.data.TextDeserializer
    classification.data.TextDataSource
    classification.data.TextCSVDataSource
    classification.data.TextJSONDataSource
    classification.data.TextDataFrameDataSource
    classification.data.TextParquetDataSource
    classification.data.TextHuggingFaceDatasetDataSource
    classification.data.TextListDataSource

Question Answering
__________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~question_answering.model.QuestionAnsweringTask
    ~question_answering.data.QuestionAnsweringData

    question_answering.data.QuestionAnsweringBackboneState
    question_answering.data.QuestionAnsweringCSVDataSource
    question_answering.data.QuestionAnsweringDataSource
    question_answering.data.QuestionAnsweringDictionaryDataSource
    question_answering.data.QuestionAnsweringFileDataSource
    question_answering.data.QuestionAnsweringJSONDataSource
    question_answering.data.QuestionAnsweringOutputTransform
    question_answering.data.QuestionAnsweringInputTransform
    question_answering.data.SQuADDataSource


Summarization
_____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.summarization.model.SummarizationTask
    ~seq2seq.summarization.data.SummarizationData

    seq2seq.summarization.data.SummarizationInputTransform

Translation
___________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.translation.model.TranslationTask
    ~seq2seq.translation.data.TranslationData

    seq2seq.translation.data.TranslationInputTransform

General Seq2Seq
_______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.core.model.Seq2SeqTask
    ~seq2seq.core.data.Seq2SeqData
    ~seq2seq.core.finetuning.Seq2SeqFreezeEmbeddings

    seq2seq.core.data.Seq2SeqBackboneState
    seq2seq.core.data.Seq2SeqCSVDataSource
    seq2seq.core.data.Seq2SeqDataSource
    seq2seq.core.data.Seq2SeqFileDataSource
    seq2seq.core.data.Seq2SeqJSONDataSource
    seq2seq.core.data.Seq2SeqOutputTransform
    seq2seq.core.data.Seq2SeqInputTransform
    seq2seq.core.data.Seq2SeqSentencesDataSource
    seq2seq.core.metrics.BLEUScore
    seq2seq.core.metrics.RougeBatchAggregator
    seq2seq.core.metrics.RougeMetric
