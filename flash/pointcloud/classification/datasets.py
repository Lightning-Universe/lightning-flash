import os

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE

if _POINTCLOUD_AVAILABLE:
    from open3d.ml.datasets import ShapeNet

_CLASSIFICATION_DATASET = FlashRegistry("dataset")


def executor(download_script, preprocess_script, dataset_path, name):
    if not os.path.exists(os.path.join(dataset_path, name)):
        os.system(f'bash -c "bash <(curl -s {download_script}) {dataset_path}"')
        if preprocess_script:
            os.system(f'bash -c "bash <(curl -s {preprocess_script}) {dataset_path}"')


@_CLASSIFICATION_DATASET
def shapenet(dataset_path):
    name = "ShapeNet"
    executor(
        "https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/scripts/download_datasets/download_shapenet.sh",
        None, dataset_path, name
    )
    return ShapeNet(os.path.join(dataset_path, name))


def ShapenetDataset(dataset_path):
    return _CLASSIFICATION_DATASET.get("shapenet")(dataset_path)
