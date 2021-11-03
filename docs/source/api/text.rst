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

    data.TextDeserializer
    classification.data.TextClassificationPostprocess
    classification.data.TextClassificationPreprocess
    classification.data.TextClassificationDataSource
    classification.data.TextClassificationCSVDataSource
    classification.data.TextClassificationJSONDataSource
    classification.data.TextClassificationDataFrameDataSource
    classification.data.TextClassificationParquetDataSource
    classification.data.TextClassificationHuggingFaceDatasetDataSource
    classification.data.TextClassificationListDataSource

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
    question_answering.data.QuestionAnsweringPostprocess
    question_answering.data.QuestionAnsweringPreprocess
    question_answering.data.SQuADDataSource


Summarization
_____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.summarization.model.SummarizationTask
    ~seq2seq.summarization.data.SummarizationData

    seq2seq.summarization.data.SummarizationPreprocess

Translation
___________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.translation.model.TranslationTask
    ~seq2seq.translation.data.TranslationData

    seq2seq.translation.data.TranslationPreprocess

General Seq2Seq
_______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.model.Seq2SeqTask
    ~seq2seq.data.Seq2SeqData
    ~seq2seq.finetuning.Seq2SeqFreezeEmbeddings

    seq2seq.data.Seq2SeqCSVDataSource
    seq2seq.data.Seq2SeqJSONDataSource
    seq2seq.data.Seq2SeqDataFrameDataSource
    seq2seq.data.Seq2SeqParquetDataSource
    seq2seq.data.Seq2SeqHuggingFaceDatasetDataSource
    seq2seq.data.Seq2SeqListDataSource

    seq2seq.data.Seq2SeqDataSource
    seq2seq.data.Seq2SeqPostprocess
    seq2seq.data.Seq2SeqPreprocess
    seq2seq.metrics.BLEUScore
    seq2seq.metrics.RougeBatchAggregator
    seq2seq.metrics.RougeMetric
