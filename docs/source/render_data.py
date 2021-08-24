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
import importlib
import os
import sys

from jinja2 import Environment, FileSystemLoader

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.join(_PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(_PATH_ROOT))

generated_dir = os.path.join("reference", "generated")
os.makedirs(generated_dir, exist_ok=True)

data_modules = [
    ("flash.audio.classification.data", "AudioClassificationData", "audio_classification.rst"),
    ("flash.image.classification.data", "ImageClassificationData", "image_classification.rst"),
    ("flash.image.segmentation.data", "SemanticSegmentationData", "semantic_segmentation.rst"),
]

env = Environment(loader=FileSystemLoader("./_templates/data"))

for data_module_path, data_module_name, template in data_modules:
    data_module = getattr(importlib.import_module(data_module_path), data_module_name)

    class PatchedPreprocess(data_module.preprocess_cls):
        """TODO: This is a hack to prevent default transforms form being created"""

        @staticmethod
        def _resolve_transforms(_):
            return None

    preprocess = PatchedPreprocess()
    data_sources = {
        data_source: preprocess.data_source_of_name(data_source) for data_source in preprocess.available_data_sources()
    }

    lines = env.get_template(template).render(
        data_module=f":class:`~{data_module_path}.{data_module_name}`",
        data_module_raw=data_module_name,
        data_sources=data_sources,
    )

    with open(os.path.join(generated_dir, f"{data_module_name}.rst"), "w") as f:
        f.writelines(lines)
