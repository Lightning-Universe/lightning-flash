import sys

import flash
from flash.core.data.utils import download_data
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE
from flash.image.style_transfer import StyleTransfer, StyleTransferData

if not _PYSTICHE_AVAILABLE:
    print("Please, run `pip install pystiche`")
    sys.exit(1)

download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

model = StyleTransfer.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/style_transfer_model.pt")

datamodule = StyleTransferData.from_folders(predict_folder="data/coco128/images/train2017", batch_size=4)

trainer = flash.Trainer(max_epochs=2)
trainer.predict(model, datamodule=datamodule)
