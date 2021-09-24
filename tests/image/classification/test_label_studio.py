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

import os

import pytest

from flash import _PACKAGE_ROOT
from flash.core.utilities.imports import _BAAL_AVAILABLE, _LABEL_STUDIO_ML_AVAILABLE
from flash.image.classification.integrations.labelstudio.handler import ImageClassifierAPI


# this test can't be run without forcing download in the ci
# label_studio_ml isn't released on pypi yet.
@pytest.mark.skipif(
    not (_LABEL_STUDIO_ML_AVAILABLE and _BAAL_AVAILABLE), reason="baal and label-studio-ml libraries aren't installed."
)
def test_image_classifier_api(tmpdir):

    ROOT_URL = "https://data.heartex.net/open-images"

    template = """
    <View>
        <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
            <Choice value="plane"/>
            <Choice value="car"/>
        </Choices>
    </View>
    """

    kwargs = {
        "label_config": template,
        "train_output": {"model_path": ""},
    }
    api = ImageClassifierAPI(source=__file__, **kwargs)
    api._IMAGE_CACHE_DIR = os.path.join(_PACKAGE_ROOT, "data", "label_studio")
    os.makedirs(api._IMAGE_CACHE_DIR, exist_ok=True)

    tasks = [
        {
            "id": 220,
            "data": {"image": f"{ROOT_URL}/train_0/mini/004e7d6b70fe1768.jpg"},
            "meta": {},
            "created_at": "2021-09-24T08:59:36.990723Z",
            "updated_at": "2021-09-24T08:59:36.990759Z",
            "is_labeled": False,
            "overlap": 1,
            "project": 6,
            "file_upload": 23,
            "annotations": [],
            "predictions": [],
        }
    ]
    api.predict(tasks)

    completion = {
        "id": 220,
        "annotations": [
            {
                "id": 25,
                "completed_by": {"id": 1, "email": "thomas@grid.ai", "first_name": "", "last_name": ""},
                "result": [
                    {
                        "value": {"choices": ["Car"]},
                        "id": "XLZp3i0DtU",
                        "from_name": "choice",
                        "to_name": "image",
                        "type": "choices",
                    }
                ],
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": "2021-09-24T09:05:07.200957Z",
                "updated_at": "2021-09-24T09:11:24.455226Z",
                "lead_time": 1722.724,
                "prediction": {},
                "result_count": 0,
                "task": 220,
            }
        ],
        "predictions": [
            {
                "id": 130,
                "model_version": "1632474140",
                "created_ago": "14\xa0minutes",
                "result": [
                    {
                        "from_name": "choice",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": ["Car"]},
                    }
                ],
                "score": 0.0007123351097106934,
                "cluster": None,
                "neighbors": None,
                "mislabeling": 0.0,
                "created_at": "2021-09-24T09:02:20.788030Z",
                "updated_at": "2021-09-24T09:02:20.788075Z",
                "task": 220,
            }
        ],
        "file_upload": "7c31b4a3-images.json",
        "data": {"image": f"{ROOT_URL}/train_0/mini/004e7d6b70fe1768.jpg"},
        "meta": {},
        "created_at": "2021-09-24T08:59:36.990723Z",
        "updated_at": "2021-09-24T09:11:24.432805Z",
        "project": 6,
    }

    out = api.fit([completion, completion], workdir=tmpdir)

    assert os.path.exists(out["model_path"])
