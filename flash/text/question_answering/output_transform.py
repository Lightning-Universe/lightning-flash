import collections
from typing import Any

from flash import OutputTransform
from flash.core.utilities.imports import requires


class QuestionAnsweringOutputTransform(OutputTransform):
    @requires("text")
    def uncollate(self, predicted_sentences: collections.OrderedDict) -> Any:
        uncollated_predicted_sentences = []
        for key in predicted_sentences:
            uncollated_predicted_sentences.append({key: predicted_sentences[key]})
        return uncollated_predicted_sentences
