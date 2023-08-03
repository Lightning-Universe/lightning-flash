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
import torchvision
from flash.core.serve import Composition, ModelComponent, expose
from flash.core.serve.types import BBox, Image, Label, Repeated


class ObjectDetection(ModelComponent):
    def __init__(self, model):
        self.model = model

    @expose(
        inputs={"img": Image()},
        outputs={"boxes": Repeated(BBox()), "labels": Repeated(Label("classes.txt"))},
    )
    def detect(self, img):
        img = img.permute(0, 3, 2, 1).float() / 255
        out = self.model(img)[0]
        return out["boxes"], out["labels"]


fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
composit = Composition(component=ObjectDetection(fasterrcnn))
composit.serve()
