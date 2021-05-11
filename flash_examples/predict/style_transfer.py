import sys

import flash
from flash.data.utils import download_data
from flash.utils.imports import _PYSTICHE_AVAILABLE
from flash.vision.style_transfer import StyleTransfer, StyleTransferData

if _PYSTICHE_AVAILABLE:
    import pystiche.demo
else:
    print("Please, run `pip install pystiche`")
    sys.exit(0)

download_data("http://images.cocodataset.org/zips/train2014.zip", "data")

data_module = StyleTransferData.from_folders(train_folder="data", batch_size=4)

style_image = pystiche.demo.images()["paint"].read(size=256)

model = StyleTransfer(style_image)

trainer = flash.Trainer(max_epochs=2)
trainer.fit(model, data_module)
