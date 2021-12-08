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

    classification.data.TextClassificationInputTransform
    classification.data.TextClassificationInput
    classification.data.TextClassificationCSVInput
    classification.data.TextClassificationJSONInput
    classification.data.TextClassificationDataFrameInput
    classification.data.TextClassificationParquetInput
    classification.data.TextClassificationListInput

Question Answering
__________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~question_answering.model.QuestionAnsweringTask
    ~question_answering.data.QuestionAnsweringData

    question_answering.data.QuestionAnsweringBackboneState
    question_answering.data.QuestionAnsweringCSVInput
    question_answering.data.QuestionAnsweringInput
    question_answering.data.QuestionAnsweringDictionaryInput
    question_answering.data.QuestionAnsweringFileInput
    question_answering.data.QuestionAnsweringJSONInput
    question_answering.data.QuestionAnsweringOutputTransform
    question_answering.data.QuestionAnsweringInputTransform
    question_answering.data.SQuADInput

Summarization
_____________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.summarization.model.SummarizationTask
    ~seq2seq.summarization.data.SummarizationData

Translation
___________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.translation.model.TranslationTask
    ~seq2seq.translation.data.TranslationData

General Seq2Seq
_______________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~seq2seq.core.model.Seq2SeqTask
    ~seq2seq.core.data.Seq2SeqData

    seq2seq.core.data.Seq2SeqBackboneState
    seq2seq.core.data.Seq2SeqInputBase
    seq2seq.core.data.Seq2SeqCSVInput
    seq2seq.core.data.Seq2SeqJSONInput
    seq2seq.core.data.Seq2SeqListInput
    seq2seq.core.data.Seq2SeqOutputTransform

flash.text.input
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    data.TextDeserializer
