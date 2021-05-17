import sys

import numpy as np
import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter

import flash
from flash.core.data.utils import download_data
from flash.core.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYSTICHE_AVAILABLE
from flash.image.style_transfer import StyleTransfer, StyleTransferData

if not _PYSTICHE_AVAILABLE:
    print("Please, run `pip install pystiche`")
    sys.exit(1)


class StyleTransferWriter(BasePredictionWriter):

    def __init__(self) -> None:
        super().__init__("batch")

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ) -> None:
        """
        Implement the logic to save a given batch of predictions.
        torch.save({"preds": prediction, "batch_indices": batch_indices}, "prediction_{batch_idx}.pt")
        """


download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

model = StyleTransfer.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/style_transfer_model.pt")

datamodule = StyleTransferData.from_folders(predict_folder="data/coco128/images/train2017", batch_size=4)

trainer = flash.Trainer(max_epochs=2, callbacks=StyleTransferWriter(), limit_predict_batches=1)
predictions = trainer.predict(model, datamodule=datamodule)

# display the first stylized image.
image_prediction = torch.stack(predictions[0])[0].numpy()

if _MATPLOTLIB_AVAILABLE and not flash._IS_TESTING:
    import matplotlib.pyplot as plt
    image = np.moveaxis(image_prediction, 0, 2)
    image -= image.min()
    image /= image.max()
    plt.imshow(image)
    plt.show()
