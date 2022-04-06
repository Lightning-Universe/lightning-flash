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

    classification.input.TextClassificationInput
    classification.input.TextClassificationCSVInput
    classification.input.TextClassificationJSONInput
    classification.input.TextClassificationDataFrameInput
    classification.input.TextClassificationParquetInput
    classification.input.TextClassificationListInput

Embedding
_________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~embedding.model.TextEmbedder

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

    seq2seq.core.input.Seq2SeqInputBase
    seq2seq.core.input.Seq2SeqCSVInput
    seq2seq.core.input.Seq2SeqJSONInput
    seq2seq.core.input.Seq2SeqListInput

flash.text.input
________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    input.TextDeserializer
