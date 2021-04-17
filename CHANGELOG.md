# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [0.2.3] - 2021-04-17

### Added

- Added TIMM integration as backbones ([#196](https://github.com/PyTorchLightning/lightning-flash/pull/196))

### Fixed

- Fixed `nltk.download` ([#210](https://github.com/PyTorchLightning/lightning-flash/pull/210))


## [0.2.2] - 2021-04-05

### Changed

- Switch to use `torchmetrics` ([#169](https://github.com/PyTorchLightning/lightning-flash/pull/169))
- Update lightning version to v1.2 ([#133](https://github.com/PyTorchLightning/lightning-flash/pull/133))

### Fixed

- Fixed classification softmax ([#169](https://github.com/PyTorchLightning/lightning-flash/pull/169))
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
