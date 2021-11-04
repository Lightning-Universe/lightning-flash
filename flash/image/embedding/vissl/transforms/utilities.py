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
import torch

from flash.core.data.io.input import InputDataKeys


def vissl_collate_helper(samples):
    result = []

    for batch_ele in samples:
        _batch_ele_dict = {}
        _batch_ele_dict.update(batch_ele)
        _batch_ele_dict[InputDataKeys.INPUT] = -1

        result.append(_batch_ele_dict)

    return torch.utils.data._utils.collate.default_collate(result)


def multicrop_collate_fn(samples):
    """Multi-crop collate function for VISSL integration.

    Run custom collate on a single key since VISSL transforms affect only InputDataKeys.INPUT
    """
    result = vissl_collate_helper(samples)

    inputs = [[] for _ in range(len(samples[0][InputDataKeys.INPUT]))]
    for batch_ele in samples:
        multi_crop_imgs = batch_ele[InputDataKeys.INPUT]

        for idx, crop in enumerate(multi_crop_imgs):
            inputs[idx].append(crop)

    for idx, ele in enumerate(inputs):
        inputs[idx] = torch.stack(ele)

    result[InputDataKeys.INPUT] = inputs

    return result


def simclr_collate_fn(samples):
    """Multi-crop collate function for VISSL integration.

    Run custom collate on a single key since VISSL transforms affect only InputDataKeys.INPUT
    """
    result = vissl_collate_helper(samples)

    inputs = []
    num_views = len(samples[0][InputDataKeys.INPUT])
    view_idx = 0
    while view_idx < num_views:
        for batch_ele in samples:
            imgs = batch_ele[InputDataKeys.INPUT]
            inputs.append(imgs[view_idx])

        view_idx += 1

    result[InputDataKeys.INPUT] = torch.stack(inputs)

    return result


def moco_collate_fn(samples):
    """MOCO collate function for VISSL integration.

    Run custom collate on a single key since VISSL transforms affect only InputDataKeys.INPUT
    """
    result = vissl_collate_helper(samples)

    inputs = []
    for batch_ele in samples:
        inputs.append(torch.stack(batch_ele[InputDataKeys.INPUT]))

    result[InputDataKeys.INPUT] = torch.stack(inputs).squeeze()[:, 0, :, :, :].squeeze()
    result["data_momentum"] = torch.stack(inputs).squeeze()[:, 1, :, :, :].squeeze()

    return result
