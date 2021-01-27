import os
import shutil
from glob import glob

import numpy as np

from flash.core.data import download_data


def hymenoptera_data_download(path: str, predict_size: int = 10):
    download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", path)
    predict_folder = os.path.join(path, "hymenoptera_data/predict")
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    if len(os.listdir(predict_folder)) > 0:
        return
    validation_image_paths = glob(os.path.join(path, "hymenoptera_data/val/*/*"))
    assert predict_size < len(validation_image_paths)
    indices = np.random.choice(range(len(validation_image_paths)), predict_size, replace=False)
    for index in indices:
        src = validation_image_paths[index]
        dst = os.path.join(predict_folder, src.split('/')[-1])
        shutil.copy(src, dst)
