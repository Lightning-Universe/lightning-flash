<div align="center">

<img src="docs/source/_static/images/logo.svg" width="400px">


**Your PyTorch AI Factory**

---

<p align="center">
  <a href="#getting-started">Installation</a> •
  <a href="#flash-in-3-steps">Flash in 3 Steps</a> •
  <a href="https://lightning-flash.readthedocs.io/en/stable/?badge=stable">Docs</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#community">Community</a> •
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="#license">License</a>
</p>


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-flash)](https://pypi.org/project/lightning-flash/)
[![PyPI Status](https://badge.fury.io/py/lightning-flash.svg)](https://badge.fury.io/py/lightning-flash)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/lightning-flash/badge/?version=latest)](https://lightning-flash.readthedocs.io/en/stable/?badge=stable)
![CI testing](https://github.com/PyTorchLightning/lightning-flash/workflows/CI%20testing/badge.svg?branch=master&event=push)
[![codecov](https://codecov.io/gh/PyTorchLightning/lightning-flash/branch/master/graph/badge.svg?token=oLuUr9q1vt)](https://codecov.io/gh/PyTorchLightning/lightning-flash)

</div>

---

<div align="center">
  Flash makes complex AI recipes for over 15 tasks across 7 data domains accessible to all.
  <br / >
  In a nutshell, Flash is the production grade research framework you always dreamed of but didn't have time to build.
</div>


## News

- Sept 30: [Lightning Flash now supports Meta-Learning](https://devblog.pytorchlightning.ai/lightning-flash-now-supports-meta-learning-7c0ac8b1cde7)
- Sept 9: [Lightning Flash 0.5](https://devblog.pytorchlightning.ai/flash-0-5-your-pytorch-ai-factory-81b172ff0d76)
- Jul 12: Flash Task-a-thon community sprint with 25+ community members
- Jul 1: [Lightning Flash 0.4](https://devblog.pytorchlightning.ai/lightning-flash-0-4-flash-serve-fiftyone-multi-label-text-classification-and-jit-support-97428276c06f)
- Jun 22: [Ushering in the New Age of Video Understanding with PyTorch](https://medium.com/pytorch/ushering-in-the-new-age-of-video-understanding-with-pytorch-1d85078e8015)
- May 24: [Lightning Flash 0.3](https://devblog.pytorchlightning.ai/lightning-flash-0-3-new-tasks-visualization-tools-data-pipeline-and-flash-registry-api-1e236ba9530)
- May 20: [Video Understanding with PyTorch](https://towardsdatascience.com/video-understanding-made-simple-with-pytorch-video-and-lightning-flash-c7d65583c37e)
- Feb 2: [Read our launch blogpost](https://pytorch-lightning.medium.com/introducing-lightning-flash-the-fastest-way-to-get-started-with-deep-learning-202f196b3b98)

## Getting Started

From PyPI:

```bash
pip install lightning-flash
```

See [our installation guide](https://lightning-flash.readthedocs.io/en/latest/installation.html) for more options.

## Flash in 3 Steps

### Step 1. Load your data

All data loading in Flash is performed via a `from_*` classmethod on a `DataModule`.
Which `DataModule` to use and which `from_*` methods are available depends on the task you want to perform.
For example, for image segmentation where your data is stored in folders, you would use the [`from_folders` method of the `SemanticSegmentationData` class](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html#from-folders):

```py
from flash.image import SemanticSegmentationData

dm = SemanticSegmentationData.from_folders(
    train_folder="data/CameraRGB",
    train_target_folder="data/CameraSeg",
    val_split=0.1,
    image_size=(256, 256),
    num_classes=21,
)

```

### Step 2: Configure your model

Our tasks come loaded with pre-trained backbones and (where applicable) heads.
You can view the available backbones to use with your task using [`available_backbones`](https://lightning-flash.readthedocs.io/en/latest/general/backbones.html).
Once you've chosen, create the model:

```py
from flash.image import SemanticSegmentation

print(SemanticSegmentation.available_heads())
# ['deeplabv3', 'deeplabv3plus', 'fpn', ..., 'unetplusplus']

print(SemanticSegmentation.available_backbones('fpn'))
# ['densenet121', ..., 'xception'] # + 113 models

print(SemanticSegmentation.available_pretrained_weights('efficientnet-b0'))
# ['imagenet', 'advprop']

model = SemanticSegmentation(
  head="fpn", backbone='efficientnet-b0', pretrained="advprop", num_classes=dm.num_classes)
```

### Step 3: Finetune!

```py
from flash import Trainer

trainer = Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")
trainer.save_checkpoint("semantic_segmentation_model.pt")
```

---

## PyTorch Recipes

### Make predictions with Flash!

Serve in just 2 lines.

```py
from flash.image import SemanticSegmentation

model = SemanticSegmentation.load_from_checkpoint("semantic_segmentation_model.pt")
model.serve()
```

or make predictions from raw data directly.

```py
predictions = model.predict(["data/CameraRGB/F61-1.png", "data/CameraRGB/F62-1.png"])
```

or make predictions with 2 GPUs.

```py
trainer = Trainer(accelerator='ddp', gpus=2)
dm = SemanticSegmentationData.from_folders(predict_folder="data/CameraRGB")
predictions = trainer.predict(model, dm)
```

### Flash Training Strategies

Training strategies are PyTorch SOTA Training Recipes which can be utilized with a given task.


Check out this [example](https://github.com/PyTorchLightning/lightning-flash/blob/master/flash_examples/integrations/learn2learn/image_classification_imagenette_mini.py) where the `ImageClassifier` supports 4 [Meta Learning Algorithms](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html) from [Learn2Learn](https://github.com/learnables/learn2learn).
This is particularly useful if you use this model in production and want to make sure the model adapts quickly to its new environment with minimal labelled data.

```py
model = ImageClassifier(
    backbone="resnet18",
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    training_strategy="prototypicalnetworks",
    training_strategy_kwargs={
        "epoch_length": 10 * 16,
        "meta_batch_size": 4,
        "num_tasks": 200,
        "test_num_tasks": 2000,
        "ways": datamodule.num_classes,
        "shots": 1,
        "test_ways": 5,
        "test_shots": 1,
        "test_queries": 15,
    },
)
```

In detail, the following methods are currently implemented:

* **[prototypicalnetworks](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_protonet.py)** : from Snell *et al.* 2017, [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
* **[maml](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_maml.py)** : from Finn *et al.* 2017, [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
* **[metaoptnet](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_metaoptnet.py)** : from Lee *et al.* 2019, [Meta-Learning with Differentiable Convex Optimization](https://arxiv.org/abs/1904.03758)
* **[anil](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_anil.py)** : from Raghu *et al.* 2020, [Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML](https://arxiv.org/abs/1909.09157)

### Flash Transforms


Flash includes some simple augmentations for each task by default, however, you will often want to override these and control your own augmentation recipe.
To this end, Flash supports custom transformations backed by our powerful data pipeline.
The transform requires to be passed as a dictionary of transforms where the keys are the [hook's name](https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.core.data.process.Preprocess.html?highlight=Preprocess).
This enable transforms to be applied per sample or per batch either on or off device.
It is important to note that data are being processed as a dictionary for all tasks (typically containing `input`, `target`, and `metadata`),
Therefore, you can use [`ApplyToKeys`](https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.core.data.transforms.ApplyToKeys.html#flash.core.data.transforms.ApplyToKeys) utility to apply the transform to a specific key.
Complex transforms (like MixUp) can then be implemented with ease.

The example also uses our [`merge_transforms`](https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.core.data.transforms.merge_transforms.html#flash.core.data.transforms.merge_transforms) utility to merge our custom augmentations with the default transforms for images (which handle resizing and converting to a tensor).


```py
import torch
from typing import Any
import numpy as np
import albumentations
from torchvision import transforms as T
from flash.core.data.transforms import ApplyToKeys, merge_transforms
from flash.image import ImageClassificationData
from flash.image.classification.transforms import default_transforms, AlbumentationsAdapter

def mixup(batch, alpha=1.0):
    images = batch["input"]
    targets = batch["target"].float().unsqueeze(1)

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0))

    batch["input"] = images * lam + images[perm] * (1 - lam)
    batch["target"] = targets * lam + targets[perm] * (1 - lam)
    return batch

train_transform = {
    # applied only on images as ApplyToKeys is used with `input`
    "post_tensor_transform": ApplyToKeys(
        "input", AlbumentationsAdapter(albumentations.HorizontalFlip(p=0.5))),

    # applied to the entire dictionary as `ApplyToKeys` isn't used.
    # this would be applied on GPUS !
    "per_batch_transform_on_device": mixup,

    # this would be applied on CPUS within the DataLoader workers !
    # "per_batch_transform": mixup
}
# merge the default transform for this task with new one.
train_transform = merge_transforms(default_transforms((256, 256)), train_transform)

datamodule = ImageClassificationData.from_folders(
    train_folder = "data/train",
    train_transform=train_transform,
)

```

## Flash Zero - PyTorch Recipes from the Command Line!

<div align="center">
<img src="/docs/source/_static/images/flash_zero.gif?raw=true" width="75%">
</div>

Flash Zero is a zero-code machine learning platform built
directly into lightning-flash
using the [`Lightning CLI`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).

To get started and view the available tasks, run:

```py
  flash --help
```

For example, to train an image classifier for 10 epochs with a `resnet50` backbone on 2 GPUs using your own data, you can do:

```py
  flash image_classification --trainer.max_epochs 10 --trainer.gpus 2 --model.backbone resnet50 from_folders --train_folder {PATH_TO_DATA}
```

## Kaggle examples

- [Titanic crash with Lightning⚡Flash](https://www.kaggle.com/jirkaborovec/titanic-crash-with-lightning-flash)
- [Covid detection with Lightning⚡️Flash](https://www.kaggle.com/jirkaborovec/covid-detection-with-lightning-flash)

---

## Contribute!
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases. But we're looking for incredible contributors like you to submit new tasks!

Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ) and/or read our [CONTRIBUTING](https://github.com/PyTorchLightning/lightning-flash/blob/master/.github/CONTRIBUTING.md) guidelines to get help becoming a contributor!

__Note:__ Flash is currently being tested on real-world use cases and is in active development. Please [open an issue](https://github.com/PyTorchLightning/lightning-flash/issues/new/choose) if you find anything that isn't working as expected.

---

## Community
Flash is maintained by our [core contributors](https://lightning-flash.readthedocs.io/en/latest/governance.html).

For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!

---

## Citations
We’re excited to continue the strong legacy of opensource software and have been inspired over the years by Caffe, Theano, Keras, PyTorch, torchbearer, and [fast.ai](https://arxiv.org/abs/2002.04688). When/if additional papers are written about this, we’ll be happy to cite these frameworks and the corresponding authors.

Flash leverages models from many different frameworks in order to cover such a wide range of domains and tasks. The full list of providers can be found in [our documentation](https://lightning-flash.readthedocs.io/en/latest/integrations/providers.html).

---

## License
Please observe the Apache 2.0 license that is listed in this repository.
