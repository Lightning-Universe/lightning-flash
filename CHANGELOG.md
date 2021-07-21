# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

- Added support for (input, target) style datasets (e.g. torchvision) to the from_datasets method ([#552](https://github.com/PyTorchLightning/lightning-flash/pull/552))

- Added support for `from_csv` and `from_data_frame` to `ImageClassificationData` ([#556](https://github.com/PyTorchLightning/lightning-flash/pull/556))

- Added SimCLR, SwAV, Barlow-twins pretrained weights for resnet50 backbone in ImageClassifier task ([#560](https://github.com/PyTorchLightning/lightning-flash/pull/560))

- Added support for Semantic Segmentation backbones and heads from `segmentation-models.pytorch` ([#562](https://github.com/PyTorchLightning/lightning-flash/pull/562))

- Added support for nesting of `Task` objects ([#575](https://github.com/PyTorchLightning/lightning-flash/pull/575))

- Added `PointCloudSegmentation` Task ([#566](https://github.com/PyTorchLightning/lightning-flash/pull/566))

- Added `PointCloudObjectDetection` Task ([#600](https://github.com/PyTorchLightning/lightning-flash/pull/600))

- Added a `GraphClassifier` task ([#73](https://github.com/PyTorchLightning/lightning-flash/pull/73))

- Added the option to pass `pretrained` as a string to `SemanticSegmentation` to change pretrained weights to load from `segmentation-models.pytorch` ([#587](https://github.com/PyTorchLightning/lightning-flash/pull/587))

- Added support for `field` parameter for loadng JSON based datasets in text tasks. ([#585](https://github.com/PyTorchLightning/lightning-flash/pull/585))

- Added `AudioClassificationData` and an example for classifying audio spectrograms ([#594](https://github.com/PyTorchLightning/lightning-flash/pull/594))

- Added a `SpeechRecognition` task for speech to text using Wav2Vec ([#586](https://github.com/PyTorchLightning/lightning-flash/pull/586))

### Changed

- Changed how pretrained flag works for loading weights for ImageClassifier task ([#560](https://github.com/PyTorchLightning/lightning-flash/pull/560))

- Removed bolts pretrained weights for SSL from ImageClassifier task ([#560](https://github.com/PyTorchLightning/lightning-flash/pull/560))

### Fixed


- Fixed a bug where serve sanity checking would not be triggered using the latest PyTorchLightning version ([#493](https://github.com/PyTorchLightning/lightning-flash/pull/493))
- Fixed a bug where train and validation metrics weren't being correctly computed ([#559](https://github.com/PyTorchLightning/lightning-flash/pull/559))

## [0.4.0] - 2021-06-22

### Added

- Added integration with FiftyOne ([#360](https://github.com/PyTorchLightning/lightning-flash/pull/360))
- Added `flash.serve` ([#399](https://github.com/PyTorchLightning/lightning-flash/pull/399))
- Added support for `torch.jit` to tasks where possible and documented task JIT compatibility ([#389](https://github.com/PyTorchLightning/lightning-flash/pull/389))
- Added option to provide a `Sampler` to the `DataModule` to use when creating a `DataLoader` ([#390](https://github.com/PyTorchLightning/lightning-flash/pull/390))
- Added support for multi-label text classification and toxic comments example ([#401](https://github.com/PyTorchLightning/lightning-flash/pull/401))
- Added a sanity checking feature to flash.serve ([#423](https://github.com/PyTorchLightning/lightning-flash/pull/423))

### Changed

- Split `backbone` argument to `SemanticSegmentation` into `backbone` and `head` arguments ([#412](https://github.com/PyTorchLightning/lightning-flash/pull/412))

### Fixed

- Fixed a bug where the `DefaultDataKeys.METADATA` couldn't be a dict ([#393](https://github.com/PyTorchLightning/lightning-flash/pull/393))
- Fixed a bug where the `SemanticSegmentation` task would not work as expected with finetuning callbacks ([#412](https://github.com/PyTorchLightning/lightning-flash/pull/412))
- Fixed a bug where predict batches could not be visualized with `ImageClassificationData` ([#438](https://github.com/PyTorchLightning/lightning-flash/pull/438))

## [0.3.2] - 2021-06-08

### Fixed

- Fixed a bug where `flash.Trainer.from_argparse_args` + `finetune` would not work ([#382](https://github.com/PyTorchLightning/lightning-flash/pull/382))

## [0.3.1] - 2021-06-08

### Added

- Added `deeplabv3`, `lraspp`, and `unet` backbones for the `SemanticSegmentation` task ([#370](https://github.com/PyTorchLightning/lightning-flash/pull/370))

### Changed

- Changed the installation command for extra features ([#346](https://github.com/PyTorchLightning/lightning-flash/pull/346))
- Change resize interpolation default mode to nearest ([#352](https://github.com/PyTorchLightning/lightning-flash/pull/352))

### Deprecated

- Deprecated `SemanticSegmentation` backbone names `torchvision/fcn_resnet50` and `torchvision/fcn_resnet101`, use `fc_resnet50` and `fcn_resnet101` instead ([#370](https://github.com/PyTorchLightning/lightning-flash/pull/370))

### Fixed

- Fixed `flash.Trainer.add_argparse_args` not adding any arguments ([#343](https://github.com/PyTorchLightning/lightning-flash/pull/343))
- Fixed a bug where the translation task wasn't decoding tokens properly ([#332](https://github.com/PyTorchLightning/lightning-flash/pull/332))
- Fixed a bug where huggingface tokenizers were sometimes being pickled ([#332](https://github.com/PyTorchLightning/lightning-flash/pull/332))
- Fixed issue with `KorniaParallelTransforms` to assure to share the random state between transforms ([#351](https://github.com/PyTorchLightning/lightning-flash/pull/351))
- Fixed a bug where using `val_split` with `overfit_batches` would give an infinite recursion ([#375](https://github.com/PyTorchLightning/lightning-flash/pull/375))
- Fixed a bug where some timm models were mistakenly given a `global_pool` argument ([#377](https://github.com/PyTorchLightning/lightning-flash/pull/377))
- Fixed `flash.Trainer.from_argparse_args` not passing arguments correctly ([#380](https://github.com/PyTorchLightning/lightning-flash/pull/380))


## [0.3.0] - 2021-05-20

### Added

- Added DataPipeline API ([#188](https://github.com/PyTorchLightning/lightning-flash/pull/188) [#141](https://github.com/PyTorchLightning/lightning-flash/pull/141) [#207](https://github.com/PyTorchLightning/lightning-flash/pull/207))
- Added timm integration ([#196](https://github.com/PyTorchLightning/lightning-flash/pull/196))
- Added BaseViz Callback ([#201](https://github.com/PyTorchLightning/lightning-flash/pull/201))
- Added backbone API ([#204](https://github.com/PyTorchLightning/lightning-flash/pull/204))
- Added support for Iterable auto dataset ([#227](https://github.com/PyTorchLightning/lightning-flash/pull/227))
- Added multi label support ([#230](https://github.com/PyTorchLightning/lightning-flash/pull/230))
- Added support for schedulers ([#232](https://github.com/PyTorchLightning/lightning-flash/pull/232))
- Added visualisation callback for image classification ([#228](https://github.com/PyTorchLightning/lightning-flash/pull/228))
- Added Video Classification task ([#216](https://github.com/PyTorchLightning/lightning-flash/pull/216))
- Added Dino backbone for image classification ([#259](https://github.com/PyTorchLightning/lightning-flash/pull/259))
- Added Data Sources API ([#256](https://github.com/PyTorchLightning/lightning-flash/pull/256) [#264](https://github.com/PyTorchLightning/lightning-flash/pull/264) [#272](https://github.com/PyTorchLightning/lightning-flash/pull/272))
- Refactor preprocess_cls to preprocess, add Serializer, add DataPipelineState ([#229](https://github.com/PyTorchLightning/lightning-flash/pull/229))
- Added Semantic Segmentation task ([#239](https://github.com/PyTorchLightning/lightning-flash/pull/239) [#287](https://github.com/PyTorchLightning/lightning-flash/pull/287) [#290](https://github.com/PyTorchLightning/lightning-flash/pull/290))
- Added Object detection prediction example ([#283](https://github.com/PyTorchLightning/lightning-flash/pull/283))
- Added Style Transfer task and accompanying finetuning and prediction examples ([#262](https://github.com/PyTorchLightning/lightning-flash/pull/262))
- Added a Template task and tutorials showing how to contribute a task to flash ([#306](https://github.com/PyTorchLightning/lightning-flash/pull/306))

### Changed

- Rename valid_ to val_ ([#197](https://github.com/PyTorchLightning/lightning-flash/pull/197))
- Refactor preprocess_cls to preprocess, add Serializer, add DataPipelineState ([#229](https://github.com/PyTorchLightning/lightning-flash/pull/229))

### Fixed

- Fix DataPipeline resolution in Task ([#212](https://github.com/PyTorchLightning/lightning-flash/pull/212))
- Fixed a bug where the backbone used in summarization was not correctly passed to the postprocess ([#296](https://github.com/PyTorchLightning/lightning-flash/pull/296))


## [0.2.3] - 2021-04-17

### Added

- Added TIMM integration as backbones ([#196](https://github.com/PyTorchLightning/lightning-flash/pull/196))

### Fixed

- Fixed nltk.download ([#210](https://github.com/PyTorchLightning/lightning-flash/pull/196))


## [0.2.2] - 2021-04-05

### Changed

- Switch to use `torchmetrics` ([#169](https://github.com/PyTorchLightning/lightning-flash/pull/169))

- Better support for `optimizer` and `schedulers` ([#232](https://github.com/PyTorchLightning/lightning-flash/pull/232))

- Update lightning version to v1.2 ([#133](https://github.com/PyTorchLightning/lightning-flash/pull/133))

### Fixed

- Fixed classification softmax ([#169](https://github.com/PyTorchLightning/lightning-flash/pull/169))

- Fixed a bug where loading from a local checkpoint that had `pretrained=True` without an internet connection would sometimes raise an error ([#237](https://github.com/PyTorchLightning/lightning-flash/pull/237))

- Don't download data if exists ([#157](https://github.com/PyTorchLightning/lightning-flash/pull/157))


## [0.2.1] - 2021-3-06

### Added

- Added `RetinaNet` & `backbones` to `ObjectDetector` Task ([#121](https://github.com/PyTorchLightning/lightning-flash/pull/121))
- Added .csv image loading utils ([#116](https://github.com/PyTorchLightning/lightning-flash/pull/116),
    [#117](https://github.com/PyTorchLightning/lightning-flash/pull/117),
    [#118](https://github.com/PyTorchLightning/lightning-flash/pull/118))

### Changed

- Set inputs as optional ([#109](https://github.com/PyTorchLightning/lightning-flash/pull/109))

### Fixed

- Set minimal requirements ([#62](https://github.com/PyTorchLightning/lightning-flash/pull/62))
- Fixed VGG backbone `num_features` ([#154](https://github.com/PyTorchLightning/lightning-flash/pull/154))


## [0.2.0] - 2021-02-12

### Added

- Added `ObjectDetector` Task ([#56](https://github.com/PyTorchLightning/lightning-flash/pull/56))
- Added TabNet for tabular classification ([#101](https://github.com/PyTorchLightning/lightning-flash/pull/#101))
- Added support for more backbones(mobilnet, vgg, densenet, resnext) ([#45](https://github.com/PyTorchLightning/lightning-flash/pull/45))
- Added backbones for image embedding model ([#63](https://github.com/PyTorchLightning/lightning-flash/pull/63))
- Added SWAV and SimCLR models to `imageclassifier` + backbone reorg ([#68](https://github.com/PyTorchLightning/lightning-flash/pull/68))

### Changed

- Applied transform in `FilePathDataset` ([#97](https://github.com/PyTorchLightning/lightning-flash/pull/97))
- Moved classification integration from vision root to folder ([#86](https://github.com/PyTorchLightning/lightning-flash/pull/86))

### Fixed

- Unfreeze default number of workers in datamodule ([#57](https://github.com/PyTorchLightning/lightning-flash/pull/57))
- Fixed wrong label in `FilePathDataset` ([#94](https://github.com/PyTorchLightning/lightning-flash/pull/94))

### Removed

- Removed `densenet161` duplicate in `DENSENET_MODELS` ([#76](https://github.com/PyTorchLightning/lightning-flash/pull/76))
- Removed redundant `num_features` arg from Classification model ([#88](https://github.com/PyTorchLightning/lightning-flash/pull/88))


## [0.1.0] - 2021-02-02

### Added

- Added flash_notebook examples ([#9](https://github.com/PyTorchLightning/lightning-flash/pull/9))
- Added `strategy` to `trainer.finetune` with `NoFreeze`, `Freeze`, `FreezeUnfreeze`, `UnfreezeMilestones` Callbacks([#39](https://github.com/PyTorchLightning/lightning-flash/pull/39))
- Added `SummarizationData`, `SummarizationTask` and `TranslationData`, `TranslationTask` ([#37](https://github.com/PyTorchLightning/lightning-flash/pull/37))
- Added `ImageEmbedder` ([#36](https://github.com/PyTorchLightning/lightning-flash/pull/36))
