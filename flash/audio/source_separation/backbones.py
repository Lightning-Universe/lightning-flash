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
from functools import partial

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ASTEROID_AVAILABLE
from flash.core.utilities.providers import _ASTEROID

ASTEROID_BACKBONES = FlashRegistry("backbones")


if _ASTEROID_AVAILABLE:

    import asteroid.models as ast
    from asteroid.models import BaseModel

    ASTEROID_BACKBONE_CLASS = [ast.ConvTasNet, ast.DPRNNTasNet]
    ASTEROID_CLASSES = {a.__name__.lower(): a for a in ASTEROID_BACKBONE_CLASS}

    def _load_asteroid_model(
        backbone: str,
        n_src: int = None,
        pretrained: bool = False,
        **kwargs,
    ) -> BaseModel:

        if backbone not in ASTEROID_BACKBONES:
            raise NotImplementedError(f"{backbone} is not implemented! Supported heads -> {ASTEROID_BACKBONES.keys()}")

        # For pretrained Models, backbone name must contain the dataset name
        if pretrained and backbone in ASTEROID_CLASSES:
            raise NotImplementedError(f"The following backbones are available -> {ASTEROID_BACKBONES.keys()}")

        return ast.get(backbone)(n_src, **kwargs), n_src

    for backbone in ASTEROID_CLASSES:
        ASTEROID_BACKBONES(
            partial(_load_asteroid_model, backbone=backbone),
            name=backbone,
            namespace="audio/sourceseparation",
            package="asteroid",
            providers=_ASTEROID,
        )

    def _load_pretrained_model(
        backbone: str,
        n_src: int,
        url: str,
        pretrained: bool = True,
    ) -> BaseModel:

        if backbone not in ASTEROID_BACKBONES:
            raise NotImplementedError(f"{backbone} is not implemented! Supported heads -> {ASTEROID_BACKBONES.keys()}")

        return BaseModel.from_pretrained(url), n_src

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri1mix_enhsingle_16k",
            n_src=1,
            url="JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k",
        ),
        name="convtasnet_libri1mix_enhsingle_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri2mix_sepclean_16k",
            n_src=2,
            url="JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
        ),
        name="convtasnet_libri2mix_sepclean_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri2mix_sepclean_8k",
            n_src=2,
            url="JorisCos/ConvTasNet_Libri2Mix_sepclean_8k",
        ),
        name="convtasnet_libri2mix_sepclean_8k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri2mix_sepnoisy_16k",
            n_src=2,
            url="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
        ),
        name="convtasnet_libri2mix_sepnoisy_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri2mix_sepnoisy_8k",
            n_src=2,
            url="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k",
        ),
        name="convtasnet_libri2mix_sepnoisy_8k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri3mix_sepclean_16k",
            n_src=3,
            url="JorisCos/ConvTasNet_Libri3Mix_sepclean_16k",
        ),
        name="convtasnet_libri3mix_sepclean_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri3mix_sepclean_8k",
            n_src=3,
            url="JorisCos/ConvTasNet_Libri3Mix_sepclean_8k",
        ),
        name="convtasnet_libri3mix_sepclean_8k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri3mix_sepnoisy_16k",
            n_src=3,
            url="JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k",
        ),
        name="convtasnet_libri3mix_sepnoisy_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="convtasnet_libri3mix_sepnoisy_8k",
            n_src=3,
            url="JorisCos/ConvTasNet_Libri3Mix_sepnoisy_8k",
        ),
        name="convtasnet_libri3mix_sepnoisy_8k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="dprnntasnet-ks2_libri1mix_enhsingle_16k",
            n_src=1,
            url="JorisCos/DPRNNTasNet-ks2_Libri1Mix_enhsingle_16k",
        ),
        name="dprnntasnet-ks2_libri1mix_enhsingle_16k",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="dprnntasnet-ks16_wham_sepclean",
            n_src=2,
            url="julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
        ),
        name="dprnntasnet-ks16_wham_sepclean",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )

    ASTEROID_BACKBONES(
        partial(
            _load_pretrained_model,
            backbone="dprnntasnet-ks2_wham_sepclean",
            n_src=2,
            url="mpariente/DPRNNTasNet-ks2_WHAM_sepclean",
        ),
        name="dprnntasnet-ks2_wham_sepclean",
        namespace="audio/sourceseparation",
        package="asteroid",
        providers=_ASTEROID,
    )
