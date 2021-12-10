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

    question_answering.input.QuestionAnsweringInputBase
    question_answering.input.QuestionAnsweringCSVInput
    question_answering.input.QuestionAnsweringJSONInput
    question_answering.input.QuestionAnsweringSQuADInput
    question_answering.input.QuestionAnsweringDictionaryInput
    question_answering.input_transform.QuestionAnsweringInputTransform
    question_answering.output_transform.QuestionAnsweringOutputTransform

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

    seq2seq.core.input.Seq2SeqInputBase
    seq2seq.core.input.Seq2SeqCSVInput
    seq2seq.core.input.Seq2SeqJSONInput
    seq2seq.core.input.Seq2SeqListInput
    seq2seq.core.data.Seq2SeqOutputTransform

flash.text.input
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    input.TextDeserializer
