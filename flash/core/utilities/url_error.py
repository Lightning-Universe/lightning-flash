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
import functools
import urllib.error

from pytorch_lightning.utilities import rank_zero_warn


def catch_url_error(fn):
    @functools.wraps(fn)
    def wrapper(*args, pretrained=False, **kwargs):
        try:
            return fn(*args, pretrained=pretrained, **kwargs)
        except urllib.error.URLError:
            # Hack for icevision/efficientdet to work without internet access
            if "efficientdet" in kwargs.get("head", ""):
                kwargs["pretrained_backbone"] = False
            result = fn(*args, pretrained=False, **kwargs)
            rank_zero_warn(
                "Failed to download pretrained weights for the selected backbone. The backbone has been created with"
                " `pretrained=False` instead. If you are loading from a local checkpoint, this warning can be safely"
                " ignored.",
                category=UserWarning,
            )
            return result

    return wrapper
