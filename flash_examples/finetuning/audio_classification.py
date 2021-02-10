import flash
from flash.core.finetuning import FreezeUnfreeze
from flash.audio import SpectrogramClassificationData, SpectrogramClassifier, download_ESC10, wav2spec

download_ESC10('.')

datamodule = SpectrogramClassificationData.from_folders(
    train_folder="./ESC-50-master/spectrograms/train",
    valid_folder="./ESC-50-master/spectrograms/valid",
    test_folder="./ESC-50-master/spectrograms/valid"
)