import torch

from flash.core.data import download_data
from flash.vision import ImageEmbedder

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Create an ImageEmbedder with swav trained on imagenet.
    # Check out SWAV: https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav
    embedder = ImageEmbedder(backbone="swav-imagenet", embedding_dim=128)

    # 3. Generate an embedding from an image path.
    embeddings = embedder.predict('data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg')

    # 4. Assert dimension
    assert embeddings.shape == torch.Size((1, 128))

    # 5. Create a tensor random image
    random_image = torch.randn(1, 3, 32, 32)

    # 6. Generate an embedding from this random image
    embeddings = embedder.predict(random_image)

    # 7. Assert dimension
    assert embeddings.shape == torch.Size((1, 128))
