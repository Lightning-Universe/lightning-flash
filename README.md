<div align="center">

<img src="docs/source/_static/images/logo.svg" width="400px">


**Collection of tasks for fast prototyping, baselining, finetuning and solving problems with deep learning**

---

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="https://lightning-flash.readthedocs.io/en/latest/?badge=latest">Docs</a> •
  <a href="#what-is-flash">About</a> •
  <a href="#predictions">Prediction</a> •
  <a href="#finetuning">Finetuning</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#a-general-task">General Task</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#community">Community</a> •
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="#license">License</a>
</p>


[![Stable API](https://img.shields.io/static/v1.svg?label=API&message=stable&color=green)](https://img.shields.io/static/v1.svg?label=API&message=stable&color=green)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-flash)](https://pypi.org/project/lightning-flash/)
[![PyPI Status](https://badge.fury.io/py/lightning-flash.svg)](https://badge.fury.io/py/lightning-flash)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![Discourse status](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforums.pytorchlightning.ai)](https://forums.pytorchlightning.ai/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)

[![Documentation Status](https://readthedocs.org/projects/lightning-flash/badge/?version=latest)](https://lightning-flash.readthedocs.io/en/latest/?badge=latest)
![CI testing](https://github.com/PyTorchLightning/lightning-flash/workflows/CI%20testing/badge.svg?branch=master&event=push)
[![codecov](https://codecov.io/gh/PyTorchLightning/lightning-flash/branch/master/graph/badge.svg?token=oLuUr9q1vt)](https://codecov.io/gh/PyTorchLightning/lightning-flash)

<!--
[![PyPI Status](https://pepy.tech/badge/lightning-flash)](https://pepy.tech/project/lightning-flash)
![Check Docs](https://github.com/PyTorchLightning/lightning-flash/workflows/Check%20Docs/badge.svg?branch=master&event=push)
-->

</div>

---
## News
[Read our launch blogpost](https://pytorch-lightning.medium.com/introducing-lightning-flash-the-fastest-way-to-get-started-with-deep-learning-202f196b3b98)

---

## Installation

Pip / conda

```bash
pip install lightning-flash -U
```

<details>
  <summary>Other installations</summary>

Pip from source

```bash
# with git
pip install git+https://github.com/PytorchLightning/lightning-flash.git@master

# OR from an archive
pip install https://github.com/PyTorchLightning/lightning-flash/archive/master.zip
```

From source using `setuptools`
``` bash
# clone flash repository locally
git clone https://github.com/PyTorchLightning/lightning-flash.git
cd lightning-flash
# install in editable mode
pip install -e .
```
</details>

---

## What is Flash
Flash is a framework of tasks for fast prototyping, baselining, finetuning and solving business and scientific problems with deep learning. It is focused on:

- Predictions
- Finetuning
- Task-based training

It is built for data scientists, machine learning practitioners, and applied researchers.


## Scalability
Flash is built on top of [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (by the Lightning team), which is a thin organizational layer on top of PyTorch. If you know PyTorch, you know PyTorch Lightning and Flash already!

As a result, Flash can scale up across any hardware (GPUs, TPUS) with zero changes to your code. It also has the best practices
in AI research embedded into each task so you don't have to be a deep learning PhD to leverage its power :)

### Predictions

```python

# import our libraries
from flash.text import TranslationTask

# 1. Load finetuned task
model = TranslationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/translation_model_en_ro.pt")

# 2. Translate a few sentences!
predictions = model.predict([
    "BBC News went to meet one of the project's first graduates.",
    "A recession has come as quickly as 11 months after the first rate hike and as long as 86 months.",
])
print(predictions)
```

### Finetuning

First, finetune:

```python
# import our libraries
import flash
from flash import download_data
from flash.vision import ImageClassificationData, ImageClassifier

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

# 2. Load the data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    valid_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
)

# 3. Build the model
model = ImageClassifier(num_classes=datamodule.num_classes, backbone="resnet18")

# 4. Create the trainer. Run once on data
trainer = flash.Trainer(max_epochs=1)

# 5. Finetune the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 6. Save it!
trainer.save_checkpoint("image_classification_model.pt")
```

Then use the finetuned model

```python
from flash.vision import ImageClassifier
# load the finetuned model
classifier = ImageClassifier.load_from_checkpoint('image_classification_model.pt')

# predict!
predictions = classifier.predict('data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg')
print(predictions)
```

---

## Tasks
Flash is built as a collection of community-built tasks. A task is highly opinionated and laser-focused on solving a single problem well, using state-of-the-art methods.

### Example 1: Image embedding
Flash has an Image embedding task to encodes an image into a vector of image features which can be used for anything like clustering, similarity search or classification.

<details>
  <summary>View example</summary>

```python
from flash.core.data import download_data
from flash.vision import ImageEmbedder

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

# 2. Create an ImageEmbedder with resnet50 trained on imagenet.
embedder = ImageEmbedder(backbone="resnet50", embedding_dim=128)

# 3. Generate an embedding from an image path.
embeddings = embedder.predict('data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg')

# 4. Print embeddings shape
print(embeddings.shape)
```

</details>

### Example 2: Text Summarization
Flash has a Summarization task to sum up text from a larger article into a short description.

<details>
  <summary>View example</summary>

```python
# import our libraries
import flash
from flash import download_data
from flash.text import SummarizationData, SummarizationTask

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", 'data/')

# 2. Load the data
datamodule = SummarizationData.from_files(
  train_file="data/xsum/train.csv",
  valid_file="data/xsum/valid.csv",
  test_file="data/xsum/test.csv",
  input="input",
  target="target"
)

# 3. Build the model
model = SummarizationTask()

# 4. Create the trainer. Run once on data
trainer = flash.Trainer(max_epochs=1, gpus=1, precision=16)

# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Test model
trainer.test()
```
To run the example:
```bash
python flash_examples/finetuning/summarization.py
```

</details>

### Example 3: Tabular Classification

Flash has a TabularClassification task to tackle any tabular classification problem.

<details>
  <summary>View example</summary>

To illustrate, say we want to build a model to predict if a passenger survived on the Titanic.

```python
# import our libraries
from torchmetrics.classification import Accuracy, Precision, Recall
import flash
from flash import download_data
from flash.tabular import TabularClassifier, TabularData

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

# 2. Load the data
datamodule = TabularData.from_csv(
  "./data/titanic/titanic.csv",
  test_csv="./data/titanic/test.csv",
  categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
  numerical_input=["Fare"],
  target="Survived",
  val_size=0.25,
)

# 3. Build the model
model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

# 4. Create the trainer. Run 10 times on data
trainer = flash.Trainer(max_epochs=10)

# 5. Train the model
trainer.fit(model, datamodule=datamodule)

# 6. Test model
trainer.test()

# 7. Predict!
predictions = model.predict("data/titanic/titanic.csv")
print(predictions)
```
To run the example:
```bash
python flash_examples/finetuning/tabular_data.py
```

</details>

### Example 4: Object Detection

Flash has a ObjectDetection task to identify and locate objects in images.

<details>
  <summary>View example</summary>

To illustrate, say we want to build a model on a tiny coco dataset.

```python
# import our libraries
import flash
from flash.core.data import download_data
from flash.vision import ObjectDetectionData, ObjectDetector

# 1. Download the data
# Dataset Credit: https://www.kaggle.com/ultralytics/coco128
download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

# 2. Load the Data
datamodule = ObjectDetectionData.from_coco(
  train_folder="data/coco128/images/train2017/",
  train_ann_file="data/coco128/annotations/instances_train2017.json",
  batch_size=2
)

# 3. Build the model
model = ObjectDetector(num_classes=datamodule.num_classes)

# 4. Create the trainer. Run twice on data
trainer = flash.Trainer(max_epochs=3)

# 5. Finetune the model
trainer.fit(model, datamodule)

# 6. Save it!
trainer.save_checkpoint("object_detection_model.pt")
```
To run the example:
```bash
python flash_examples/finetuning/object_detection.py
```

</details>

## A general task
Flash comes prebuilt with a task to handle a huge portion of deep learning problems.

```python
import flash
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# data
dataset = datasets.MNIST('./data_folder', download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

# task
classifier = flash.Task(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

# train
flash.Trainer().fit(classifier, DataLoader(train), DataLoader(val))
```

## Infinitely customizable

Tasks can be built in just a few minutes because Flash is built on top of PyTorch Lightning LightningModules, which
are infinitely extensible and let you train across GPUs, TPUs etc without doing any code changes.

```python
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from typing import Callable, Mapping, Sequence, Type, Union
from flash.core.classification import ClassificationTask

class LinearClassifier(ClassificationTask):
    def __init__(
        self,
        num_inputs,
        num_classes,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()

        self.linear = torch.nn.Linear(num_inputs, num_classes)

    def forward(self, x):
        return self.linear(x)

classifier = LinearClassifier()
...

```

When you reach the limits of the flexibility provided by tasks, then seamlessly transition to PyTorch Lightning which
gives you the most flexibility because it is simply organized PyTorch.

## Contribute!
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases. But we're looking for incredible contributors like you to submit new tasks!

Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A) and/or read our [CONTRIBUTING](https://github.com/PyTorchLightning/lightning-flash/blob/master/.github/CONTRIBUTING.md) guidelines to get help becoming a contributor!

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## Citations
We’re excited to continue the strong legacy of opensource software and have been inspired over the years by Caffee, Theano, Keras, PyTorch, torchbearer, and fast.ai. When/if a paper is written about this, we’ll be happy to cite these frameworks and the corresponding authors.

Flash leverages models from [torchvision](https://pytorch.org/vision/stable/index.html), [huggingface/transformers](https://huggingface.co/transformers/), and [pytorch-tabnet](https://dreamquark-ai.github.io/tabnet/) for the `vision`, `text`, and `tabular` tasks respectively. Also supports self-supervised backbones from [bolts](https://github.com/PyTorchLightning/lightning-bolts).

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition
the Lightning framework is Patent Pending.
