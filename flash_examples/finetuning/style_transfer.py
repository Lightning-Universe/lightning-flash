import sys

import flash
from flash.core.data.utils import download_data
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE

if _PYSTICHE_AVAILABLE:
    import pystiche.demo

    from flash.image.style_transfer import StyleTransfer, StyleTransferData
else:
    print("Please, run `pip install pystiche`")
    sys.exit(1)

download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

data_module = StyleTransferData.from_folders(train_folder="data/coco128/images", batch_size=4)

style_image = pystiche.demo.images()["paint"].read(size=256)

model = StyleTransfer(style_image)

trainer = flash.Trainer(max_epochs=2)
trainer.fit(model, data_module)

trainer.save_checkpoint("style_transfer_model.pt")
