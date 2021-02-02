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
import pandas as pd
import pytorch_lightning as pl

from flash.tabular import TabularClassifier, TabularData

TEST_DF_1 = pd.DataFrame(
    data={
        "category": ["a", "b", "c", "a", None, "c"],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)


def test_classification(tmpdir):

    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_1.copy()
    test_df = TEST_DF_1.copy()
    data = TabularData.from_df(
        train_df,
        categorical_input=["category"],
        numerical_input=["scalar_a", "scalar_b"],
        target="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=2,
    )
    model = TabularClassifier(num_features=3, num_classes=2, embedding_sizes=data.emb_sizes)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)
