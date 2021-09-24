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

from sentence_transformers import SentenceTransformer
from typing import List, Union
from numpy import ndarray
from torch import  Tensor
class SentenceEmbeddings():
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model=SentenceTransformer(model_name_or_path=self.model_name)
        
    def return_embeddings(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        return self.model.encode(sentences=sentences,
                            batch_size=batch_size,
                            show_progress_bar=show_progress_bar,
                            output_value=output_value,
                            convert_to_numpy=convert_to_numpy,
                            convert_to_tensor=convert_to_tensor,
                            device=device,
                            normalize_embeddings=normalize_embeddings)
        