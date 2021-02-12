# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [Unreleased] - 2021-MM-DD

### Added



### Changed



### Fixed



### Removed




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
