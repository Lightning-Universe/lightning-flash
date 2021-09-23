# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch

import flash
from flash.core.classification import Probabilities
from flash.core.utilities.imports import _BAAL_AVAILABLE, _LABEL_STUDIO_ML_AVAILABLE
from flash.image import ImageClassificationData, ImageClassifier

if _BAAL_AVAILABLE:
    from baal.active.heuristics import BALD

if _LABEL_STUDIO_ML_AVAILABLE:
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.utils import get_choice, get_single_tag_keys, is_skipped

else:

    class LabelStudioMLBase:
        pass


class FlashImageClassifierLabelStudioML(LabelStudioMLBase):

    _IMAGE_CACHE_DIR: str

    def __init__(self, source: str, **kwargs):
        super().__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "Choices", "Image"
        )

        num_classes = len(self.classes)

        head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, num_classes),
        )

        self.model = ImageClassifier(
            backbone="resnet18", num_classes=num_classes, head=head, serializer=Probabilities()
        )

        self.heuristic = BALD()

        self._IMAGE_CACHE_DIR = os.path.join(os.path.dirname(source), "image_cache")
        os.makedirs(self._IMAGE_CACHE_DIR, exist_ok=True)

    def predict(self, tasks, **kwargs):
        image_urls = [self._firstv(task["data"]) for task in tasks]
        predict_files = [self._download_image(url) for url in image_urls]

        datamodule = ImageClassificationData.from_files(predict_files=predict_files)
        trainer = flash.Trainer(max_epochs=1, gpus=int(torch.cuda.is_available()))
        predictions = trainer.predict(self.model, datamodule=datamodule)
        predictions = torch.tensor(predictions)
        predicted_label_indices = predictions[0].argmax(-1)
        predicted_scores = self.heuristic.get_uncertainties(predictions)

        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.classes[idx]
            # prediction result for the single task
            result = [
                {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "choices",
                    "value": {"choices": [predicted_label]},
                }
            ]

            # expand predictions with their scores for all tasks
            predictions.append({"result": result, "score": float(score)})

        return predictions

    def fit(self, completions, workdir=None, batch_size=32, num_epochs=10, **kwargs):
        image_urls, image_classes = [], []
        for completion in completions:

            if is_skipped(completion):
                continue
            image_urls.append(completion["data"][self.value])
            image_classes.append(get_choice(completion))

        train_files = [self._download_image(url, url_class) for url, url_class in zip(image_urls, image_classes)]
        class_to_id = {c: id for id, c in enumerate(np.unique(image_classes))}
        train_targets = np.vectorize(class_to_id.get)(image_classes)
        datamodule = ImageClassificationData.from_files(train_files=train_files, train_targets=train_targets)
        trainer = flash.Trainer(max_epochs=1, gpus=int(torch.cuda.is_available()))
        trainer.fit(self.model, datamodule=datamodule)
        model_path = os.path.join(".", "model.pt")
        trainer.save_checkpoint(model_path)
        return {"model_path": model_path}

    @staticmethod
    def _firstv(d: Dict) -> Any:
        return next(iter(d.values()))

    def _download_image(self, url, url_class: Optional[str] = None):
        is_local_file = url.startswith("/data")
        if is_local_file:
            return url
        else:
            root = self._IMAGE_CACHE_DIR
            if url_class:
                root = os.path.join(root, url_class)
                if not os.path.exists(root):
                    os.makedirs(root)
            _hash = hashlib.md5(url.encode()).hexdigest()
            cached_file = os.path.join(root, f"{_hash}.jpg")
            if os.path.exists(cached_file):
                return cached_file
            else:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(cached_file, mode="wb") as fout:
                    fout.write(r.content)
        return cached_file
