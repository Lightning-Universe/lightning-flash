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
from typing import Callable

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ASTEROID_AVAILABLE

if _ASTEROID_AVAILABLE:
	import asteroid.models as ast

	ASTEROID_BACKBONE_CLASS = [
		ast.ConvTasNet, ast.DPRNNTasNet
	]
	ASTEROID_BACKBONES = {a.__name__.lower(): a for a in ASTEROID_MODEL_CLASS}

ASTEROID_BACKBONES = FlashRegistry("backbones")

if _ASTEROID_AVAILABLE:

	def _load_asteroid_model(
		backbone: str,
		n_src: int = 2,
		pretrained: bool = False,
		**kwargs,

	) -> Callable:

		if backbone not in ASTEROID_BACKBONES:
			raise NotImplementedError(f"{backbone} is not implemented! Supported heads -> {ASTEROID_BACKBONES.keys()}")

		# Pretraining loading logic goes here 
		if pretrained:
			return ast.get(backbone).from_pretrained("<SOME LINK GOES HERE>")
		
		return ast.get(backbone)(n_src,**kwargs)

