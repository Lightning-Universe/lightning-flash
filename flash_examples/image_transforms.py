from torchvision import transforms as T

from flash.core.data.utils import download_data
from flash.image import ImageClassificationData
from flash.image.classification.transforms import default_transforms, merge_transforms

image_size = (256, 256)
default_image_transforms = default_transforms(image_size)

post_tensor_transform = T.Compose([
    T.RandomHorizontalFlip(), T.ColorJitter(),
    T.RandomAutocontrast(), T.RandomPerspective()
])

new_transforms = merge_transforms(default_image_transforms, {"post_tensor_transform": post_tensor_transform})

download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    train_transform=new_transforms
)
