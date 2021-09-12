<div align="center">

<img src="docs/source/_static/images/logo.svg" width="400px">


**The PyTorch AI Factory**

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="https://lightning-flash.readthedocs.io/en/stable/?badge=stable">Docs</a> •
  <a href="#what-is-flash--setting-up-your-recipe">About</a> •
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

<!--
[![PyPI Status](https://pepy.tech/badge/lightning-flash)](https://pepy.tech/project/lightning-flash)
![Check Docs](https://github.com/PyTorchLightning/lightning-flash/workflows/Check%20Docs/badge.svg?branch=master&event=push)
-->

</div>

<div align="center">
<a href="https://lightning-flash.readthedocs.io/en/stable">
<img src="https://pl-flash-data.s3.amazonaws.com/assets/banner.gif">
</a>
</div>

---

## News

- Jul 12: Flash Task-a-thon community sprint with 25+ community members
- Jul 1: [Lightning Flash 0.4](https://devblog.pytorchlightning.ai/lightning-flash-0-4-flash-serve-fiftyone-multi-label-text-classification-and-jit-support-97428276c06f)
- Jun 22: [Ushering in the New Age of Video Understanding with PyTorch](https://medium.com/pytorch/ushering-in-the-new-age-of-video-understanding-with-pytorch-1d85078e8015)
- May 24: [Lightning Flash 0.3](https://devblog.pytorchlightning.ai/lightning-flash-0-3-new-tasks-visualization-tools-data-pipeline-and-flash-registry-api-1e236ba9530)
- May 20: [Video Understanding with PyTorch](https://towardsdatascience.com/video-understanding-made-simple-with-pytorch-video-and-lightning-flash-c7d65583c37e)
- Feb 2: [Read our launch blogpost](https://pytorch-lightning.medium.com/introducing-lightning-flash-the-fastest-way-to-get-started-with-deep-learning-202f196b3b98)

__Note:__ Flash is currently being tested on real-world use cases and is in active development. Please [open an issue](https://github.com/PyTorchLightning/lightning-flash/issues/new/choose) if you find anything that isn't working as expected.

---

## Installation

Pip / conda

```bash
pip install lightning-flash
```

See [our installation guide](https://lightning-flash.readthedocs.io/en/latest/installation.html) for more options.

---

## What is Flash / Setting up Your Recipe

Flash is an AI factory for PyTorch.
It enables you to easily configure and run complex AI recipes for over 15 tasks across 7 different data domains.
To use Flash you need two things: some data, and a task you'd like to perform.
You can browse the tasks that we support and filter by useful tags like data type from [our documentation](https://lightning-flash.readthedocs.io/en/stable/).

### 1. Load your data

### 2. Configure your transforms

### 3. Configure your model

### 4. Define your pipeline

### 5. Run it!

---

## Contribute!
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases. But we're looking for incredible contributors like you to submit new tasks!

Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ) and/or read our [CONTRIBUTING](https://github.com/PyTorchLightning/lightning-flash/blob/master/.github/CONTRIBUTING.md) guidelines to get help becoming a contributor!

---

## Community
Flash is maintained by our [core contributors](https://lightning-flash.readthedocs.io/en/latest/governance.html).

For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)!

---

## Citations
We’re excited to continue the strong legacy of opensource software and have been inspired over the years by Caffe, Theano, Keras, PyTorch, torchbearer, and fast.ai. When/if a paper is written about this, we’ll be happy to cite these frameworks and the corresponding authors.

Flash leverages models from [torchvision](https://pytorch.org/vision/stable/index.html), [huggingface/transformers](https://huggingface.co/transformers/), [timm](https://github.com/rwightman/pytorch-image-models), [open3d-ml](https://github.com/intel-isl/Open3D-ML) for pointcloud, [pytorch-tabnet](https://dreamquark-ai.github.io/tabnet/), and [asteroid](https://github.com/asteroid-team/asteroid) for the `vision`, `text`, `tabular`, and `audio` tasks respectively. Also supports self-supervised backbones from [bolts](https://github.com/PyTorchLightning/lightning-bolts).

---

## License
Please observe the Apache 2.0 license that is listed in this repository.
