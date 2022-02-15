# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.7.0] - 2022-15-02

### Added

- Added support for multi-label, space delimited, targets ([#1076](https://github.com/PyTorchLightning/lightning-flash/pull/1076))
- Added support for tabular classification / regression backbones from PyTorch Tabular ([#1098](https://github.com/PyTorchLightning/lightning-flash/pull/1098))
- Added Flash zero support for tabular regression ([#1098](https://github.com/PyTorchLightning/lightning-flash/pull/1098))
- Added support for COCO annotations with non-default keypoint labels to `KeypointDetectionData.from_coco` ([#1102](https://github.com/PyTorchLightning/lightning-flash/pull/1102))
- Added support for `from_csv` and `from_data_frame` to `VideoClassificationData` ([#1117](https://github.com/PyTorchLightning/lightning-flash/pull/1117))
- Added support for `SemanticSegmentationData.from_folders` where mask files have different extensions to the image files ([#1130](https://github.com/PyTorchLightning/lightning-flash/pull/1130))
- Added `FlashRegistry` of Available Heads for `flash.image.ImageClassifier` ([#1152](https://github.com/PyTorchLightning/lightning-flash/pull/1152))
- Added support for `ObjectDetectionData.from_files` ([#1154](https://github.com/PyTorchLightning/lightning-flash/pull/1154))
- Added support for passing the `Output` object (or a string e.g. `"labels"`) to the `flash.Trainer.predict` method ([#1157](https://github.com/PyTorchLightning/lightning-flash/pull/1157))
- Added support for passing the `TargetFormatter` object to `from_*` methods for classification to override target handling ([#1171](https://github.com/PyTorchLightning/lightning-flash/pull/1171))

### Changed

- Changed `Wav2Vec2Processor` to `AutoProcessor` and seperate it from backbone [optional] ([#1075](https://github.com/PyTorchLightning/lightning-flash/pull/1075))
- Renamed `ClassificationInput` to `ClassificationInputMixin` ([#1116](https://github.com/PyTorchLightning/lightning-flash/pull/1116))
- Changed the default `learning_rate` for all tasks to be `None`, corresponding to the default for your chosen optimizer ([#1172](https://github.com/PyTorchLightning/lightning-flash/pull/1172))

### Fixed

- Fixed a bug when not explicitly passing `embedding_sizes` to the `TabularClassifier` and `TabularRegressor` tasks ([#1067](https://github.com/PyTorchLightning/lightning-flash/pull/1067))
- Fixed a bug where under some circumstances transforms would not get called ([#1072](https://github.com/PyTorchLightning/lightning-flash/pull/1072))
- Fixed a bug where prediction would sometimes give the wrong number of outputs ([#1077](https://github.com/PyTorchLightning/lightning-flash/pull/1077))
- Fixed a bug where passing the `val_split` to the `DataModule` would not have the desired effect ([#1079](https://github.com/PyTorchLightning/lightning-flash/pull/1079))
- Fixed a bug where passing `predict_data_frame` to `ImageClassificationData.from_data_frame` raised an error ([#1088](https://github.com/PyTorchLightning/lightning-flash/pull/1088))
- Fixed a bug where segmentation files / masks were loaded with an inconsistent ordering ([#1094](https://github.com/PyTorchLightning/lightning-flash/pull/1094))
- Fixed a bug with `AudioClassificationData.from_numpy` ([#1096](https://github.com/PyTorchLightning/lightning-flash/pull/1096))
- Fixed a bug when using `SpeechRecognitionData.from_files` for training / validating / testing ([#1097](https://github.com/PyTorchLightning/lightning-flash/pull/1097))
- Fixed a bug when using `SpeechRecognitionData.from_csv` or `from_json` when predicting without targets ([#1097](https://github.com/PyTorchLightning/lightning-flash/pull/1097))
- Fixed a bug where `SpeechRecognitionData.from_datasets` did not work as expected ([#1097](https://github.com/PyTorchLightning/lightning-flash/pull/1097))
- Fixed a bug where loading data for prediction with `SemanticSegmentationData.from_folders` raised an error ([#1101](https://github.com/PyTorchLightning/lightning-flash/pull/1101))
- Fixed a bug when passing a `predict_folder` argument to `from_coco` / `from_voc` / `from_via` in IceVision tasks ([#1102](https://github.com/PyTorchLightning/lightning-flash/pull/1102))
- Fixed `ObjectDetectionData.from_voc` and `ObjectDetectionData.from_via` ([#1102](https://github.com/PyTorchLightning/lightning-flash/pull/1102))
- Fixed a bug where `InstanceSegmentationData.from_coco` would raise an error if not using file-based masks ([#1102](https://github.com/PyTorchLightning/lightning-flash/pull/1102))
- Fixed `InstanceSegmentationData.from_voc` ([#1102](https://github.com/PyTorchLightning/lightning-flash/pull/1102))
- Fixed a bug when loading tabular data for prediction without a target field / column ([#1114](https://github.com/PyTorchLightning/lightning-flash/pull/1114))
- Fixed a bug when loading prediction data for graph classification without targets ([#1121](https://github.com/PyTorchLightning/lightning-flash/pull/1121))
- Fixed a bug where loading Seq2Seq data for prediction would not work if the target field was not present ([#1128](https://github.com/PyTorchLightning/lightning-flash/pull/1128))
- Fixed a bug where `from_fiftyone` classmethods did not work correctly with a `predict_dataset` ([#1136](https://github.com/PyTorchLightning/lightning-flash/pull/1136))
- Fixed a bug where the `labels` property would return `None` when using `ObjectDetectionData.from_fiftyone` ([#1136](https://github.com/PyTorchLightning/lightning-flash/pull/1136))
- Fixed a bug where `TabularData` would not work correctly with no categorical variables ([#1144](https://github.com/PyTorchLightning/lightning-flash/pull/1144))
- Fixed a bug where loading `TabularForecastingData` for prediction would only yield a single sample per series ([#1149](https://github.com/PyTorchLightning/lightning-flash/pull/1149))
- Fixed a bug where backbones for the `ObjectDetector`, `KeypointDetector`, and `InstanceSegmentation` tasks were not always frozen correctly when finetuning ([#1163](https://github.com/PyTorchLightning/lightning-flash/pull/1163))
- Fixed a bug where `DataModule.multi_label` would sometimes be `None` when it had been inferred to be `False` ([#1165](https://github.com/PyTorchLightning/lightning-flash/pull/1165))

### Removed

- Removed the `Seq2SeqData` base class (use `TranslationData` or `SummarizationData` directly) ([#1128](https://github.com/PyTorchLightning/lightning-flash/pull/1128))
- Removed the ability to attach the `Output` object directly to the model ([#1157](https://github.com/PyTorchLightning/lightning-flash/pull/1157))

## [0.6.0] - 2021-13-12

### Added

- Added `TextEmbedder` task ([#996](https://github.com/PyTorchLightning/lightning-flash/pull/996))
- Added predict_kwargs in `ObjectDetector`, `InstanceSegmentation`, `KeypointDetector` ([#990](https://github.com/PyTorchLightning/lightning-flash/pull/990))
- Added backbones for `GraphClassifier` ([#592](https://github.com/PyTorchLightning/lightning-flash/pull/592))
- Added `GraphEmbedder` task ([#592](https://github.com/PyTorchLightning/lightning-flash/pull/592))
- Added support for comma delimited multi-label targets to the `ImageClassifier` ([#997](https://github.com/PyTorchLightning/lightning-flash/pull/997))
- Added `datapipeline_state` on dataset creation within the `from_*` methods from the `DataModule` ([#1018](https://github.com/PyTorchLightning/lightning-flash/pull/1018))

### Changed

- Changed `DataSource` to `Input` ([#929](https://github.com/PyTorchLightning/lightning-flash/pull/929))
- Changed `Preprocess` to `InputTransform` ([#951](https://github.com/PyTorchLightning/lightning-flash/pull/951))
- Changed classes named `*Serializer` and properties / variables named `serializer` to be `*Output` and `output` respectively ([#927](https://github.com/PyTorchLightning/lightning-flash/pull/927))
- Changed `Postprocess` to `OutputTransform` ([#942](https://github.com/PyTorchLightning/lightning-flash/pull/942))
- Changed loading of RGBA images to drop alpha channel by default ([#946](https://github.com/PyTorchLightning/lightning-flash/pull/946))
- Updated `FlashFinetuning` callback to use separate hooks that lets users use the freezing logic provided out-of-the-box from flash, route FlashFinetuning through a registry. ([#830](https://github.com/PyTorchLightning/lightning-flash/pull/830))
- Changed the `SpeechRecognition` task to use `AutoModelForCTC` rather than just `Wav2Vec2ForCTC` ([#874](https://github.com/PyTorchLightning/lightning-flash/pull/874))
- Changed the `Deserializer` to subclass `ServeInput` ([#1013](https://github.com/PyTorchLightning/lightning-flash/pull/1013))
- Added `Output` suffix to `Preds`, `FiftyOneDetectionLabels`, `SegmentationLabels`, `FiftyOneDetectionLabels`, `DetectionLabels`, `Classes`, `FiftyOneLabels`, `Labels`, `Logits`, `Probabilities` ([#1011](https://github.com/PyTorchLightning/lightning-flash/pull/1011))
- Changed `from_files` and `from_folders` from `ObjectDetectionData`, `InstanceSegmentationData`, `KeypointDetectionData` to support only the `predicting` stage ([#1018](https://github.com/PyTorchLightning/lightning-flash/pull/1018))
- Changed `Image Classification Task` to use the new DataModule API ([#1025](https://github.com/PyTorchLightning/pytorch-lightning/pull/1025))

### Deprecated

- Deprecated `flash.core.data.process.Serializer` in favour of `flash.core.data.io.output.Output` ([#927](https://github.com/PyTorchLightning/lightning-flash/pull/927))
- Deprecated `Task.serializer` in favour of `Task.output` ([#927](https://github.com/PyTorchLightning/lightning-flash/pull/927))
- Deprecated `flash.text.seq2seq.core.metrics` in favour of `torchmetrics[text]` ([#648](https://github.com/PyTorchLightning/lightning-flash/pull/648))
- Deprecated `flash.core.data.data_source.DefaultDataKeys` in favour of `flash.DataKeys` ([#929](https://github.com/PyTorchLightning/lightning-flash/pull/929))
- Deprecated `data_source` argument to `flash.Task.predict` in favour of `input` ([#929](https://github.com/PyTorchLightning/lightning-flash/pull/929))

### Fixed

- Fixed a bug where using image classification with DDP spawn would trigger an infinite recursion ([#969](https://github.com/PyTorchLightning/lightning-flash/pull/969))
- Fixed a bug where Flash could not be used with IceVision 0.11.0 ([#989](https://github.com/PyTorchLightning/lightning-flash/pull/989))
- Fixed a bug where backbone weights were sometimes not frozen correctly ([#992](https://github.com/PyTorchLightning/lightning-flash/pull/992))
- Fixed a bug where translation metrics were not computed correctly ([#992](https://github.com/PyTorchLightning/lightning-flash/pull/992))
- Fixed a bug where additional `DataModule` keyword arguments could not be configured with Flash Zero for some tasks ([#994](https://github.com/PyTorchLightning/lightning-flash/pull/994))
- Fixed a bug where the TabularForecaster would not work with some versions of pandas ([#995](https://github.com/PyTorchLightning/lightning-flash/pull/995))

### Removed

- Removed `OutputMapping` ([#939](https://github.com/PyTorchLightning/lightning-flash/pull/939))
- Removed `Output.enable` and `Output.disable` ([#939](https://github.com/PyTorchLightning/lightning-flash/pull/939))
- Removed `OutputTransform.save_sample` and `save_data` hooks ([#948](https://github.com/PyTorchLightning/lightning-flash/pull/948))
- Removed InputTransform `pre_tensor_transform`, `to_tensor_transform`, `post_tensor_transform` hooks in favour of `per_sample_transform` ([#1010](https://github.com/PyTorchLightning/lightning-flash/pull/1010))
- Removed `Task.predict`, use `Trainer.predict` instead ([#1030](https://github.com/PyTorchLightning/lightning-flash/pull/1030))
- Removed the `backbone` argument from `TextClassificationData`, it is now sufficient to only provide a `backbone` argument to the `TextClassifier` ([#1022](https://github.com/PyTorchLightning/lightning-flash/pull/1022))
- Removed support for the `serve_sanity_check` argument in `flash.Trainer` ([#1062](https://github.com/PyTorchLightning/lightning-flash/pull/1062))

## [0.5.2] - 2021-11-05

### Added

- Added a `TabularForecaster` task based on PyTorch Forecasting ([#647](https://github.com/PyTorchLightning/lightning-flash/pull/647))
- Added a `TabularRegressor` task ([#892](https://github.com/PyTorchLightning/lightning-flash/pull/892))

### Fixed

- Fixed a bug where test metrics were not logged correctly with active learning ([#879](https://github.com/PyTorchLightning/lightning-flash/pull/879))
- Fixed a bug where validation metrics could be aggregated together with test metrics in some cases ([#900](https://github.com/PyTorchLightning/lightning-flash/pull/900))
- Fixed a bug where the latest versions of torchmetrics and Lightning Flash could not be installed together ([#902](https://github.com/PyTorchLightning/lightning-flash/pull/902))
- Fixed compatibility with PyTorch-Lightning 1.5 ([#933](https://github.com/PyTorchLightning/lightning-flash/pull/933))


## [0.5.1] - 2021-10-26

### Added

- Added `LabelStudio` integration ([#554](https://github.com/PyTorchLightning/lightning-flash/pull/554))
- Added support `learn2learn` training_strategy for `ImageClassifier` ([#737](https://github.com/PyTorchLightning/lightning-flash/pull/737))
- Added `vissl` training_strategies for `ImageEmbedder` ([#682](https://github.com/PyTorchLightning/lightning-flash/pull/682))
- Added support for `from_data_frame` to `TextClassificationData` ([#785](https://github.com/PyTorchLightning/lightning-flash/pull/785))
- Added `FastFace` integration ([#606](https://github.com/PyTorchLightning/lightning-flash/pull/606))
- Added support for `from_lists` to `TextClassificationData` ([#805](https://github.com/PyTorchLightning/lightning-flash/pull/805))

### Changed

- Changed the default `num_workers` on linux to `0` (matching the default for other OS) ([#759](https://github.com/PyTorchLightning/lightning-flash/pull/759))
- Optimizer and LR Scheduler registry are used to get the respective inputs to the Task using a string (or a callable). ([#777](https://github.com/PyTorchLightning/lightning-flash/pull/777))

### Fixed

- Fixed a bug where additional kwargs (e.g. sampler) passed to tabular data would be ignored ([#792](https://github.com/PyTorchLightning/lightning-flash/pull/792))
- Fixed a bug where loading text data with additional non-numeric columns (not input or target) would give an error ([#888](https://github.com/PyTorchLightning/lightning-flash/pull/888))


## [0.5.0] - 2021-09-07

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
- Added Flash Zero, a zero code command line ML platform built with flash ([#611](https://github.com/PyTorchLightning/lightning-flash/pull/611))
- Added support for `.npy` and `.npz` files to `ImageClassificationData` and `AudioClassificationData` ([#651](https://github.com/PyTorchLightning/lightning-flash/pull/651))
- Added support for `from_csv` to the `AudioClassificationData` ([#651](https://github.com/PyTorchLightning/lightning-flash/pull/651))
- Added option to pass a `resolver` to the `from_csv` and `from_pandas` methods of `ImageClassificationData`, which is used to resolve filenames given IDs ([#651](https://github.com/PyTorchLightning/lightning-flash/pull/651))
- Added integration with IceVision for the `ObjectDetector` ([#608](https://github.com/PyTorchLightning/lightning-flash/pull/608))
- Added keypoint detection task ([#608](https://github.com/PyTorchLightning/lightning-flash/pull/608))
- Added instance segmentation task ([#608](https://github.com/PyTorchLightning/lightning-flash/pull/608))
- Added Torch ORT support to Transformer based tasks ([#667](https://github.com/PyTorchLightning/lightning-flash/pull/667))
- Added support for flash zero with the `InstanceSegmentation` and `KeypointDetector` tasks ([#672](https://github.com/PyTorchLightning/lightning-flash/pull/672))
- Added support for `in_chans` argument to the flash ResNet to control the expected number of input channels ([#673](https://github.com/PyTorchLightning/lightning-flash/pull/673))
- Added a `QuestionAnswering` task for extractive question answering ([#607](https://github.com/PyTorchLightning/lightning-flash/pull/607))
- Added automatic unwrapping of IceVision prediction objects ([#727](https://github.com/PyTorchLightning/lightning-flash/pull/727))
- Added support for the `ObjectDetector` with FiftyOne ([#727](https://github.com/PyTorchLightning/lightning-flash/pull/727))
- Added support for MP3 files to the `SpeechRecognition` task with librosa ([#726](https://github.com/PyTorchLightning/lightning-flash/pull/726))
- Added support for `from_numpy` and `from_tensors` to `AudioClassificationData` ([#745](https://github.com/PyTorchLightning/lightning-flash/pull/745))

### Changed

- Changed how pretrained flag works for loading weights for ImageClassifier task ([#560](https://github.com/PyTorchLightning/lightning-flash/pull/560))
- Removed bolts pretrained weights for SSL from ImageClassifier task ([#560](https://github.com/PyTorchLightning/lightning-flash/pull/560))
- Changed the behaviour of the `sampler` argument of the `DataModule` to take a `Sampler` type rather than instantiated object ([#651](https://github.com/PyTorchLightning/lightning-flash/pull/651))
- Changed arguments to `ObjectDetector`, use `head` instead of `model` and append `_fpn` to the backbone name instead of the `fpn` argument ([#608](https://github.com/PyTorchLightning/lightning-flash/pull/608))

### Fixed

- Fixed a bug where serve sanity checking would not be triggered using the latest PyTorchLightning version ([#493](https://github.com/PyTorchLightning/lightning-flash/pull/493))
- Fixed a bug where train and validation metrics weren't being correctly computed ([#559](https://github.com/PyTorchLightning/lightning-flash/pull/559))
- Fixed a bug where an uncaught ValueError could be raised when checking if a module is available ([#615](https://github.com/PyTorchLightning/lightning-flash/pull/615))
- Fixed a bug where some tasks were not compatible with PyTorch 1.7 due to use of `torch.jit.isinstance` ([#611](https://github.com/PyTorchLightning/lightning-flash/pull/611))
- Fixed a bug where custom samplers would not be properly forwarded to the data loader ([#651](https://github.com/PyTorchLightning/lightning-flash/pull/651))
- Fixed a bug where it was not possible to pass no metrics to the `ImageClassifier` or `TestClassifier` ([#660](https://github.com/PyTorchLightning/lightning-flash/pull/660))
- Fixed a bug where `drop_last` would be set to True during prediction and testing ([#671](https://github.com/PyTorchLightning/lightning-flash/pull/671))
- Fixed a bug where flash was not compatible with pytorch-lightning >= 1.4.3 ([#690](https://github.com/PyTorchLightning/lightning-flash/pull/690))

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
