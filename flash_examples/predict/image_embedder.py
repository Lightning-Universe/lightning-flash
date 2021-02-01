import torch

from flash.vision import ImageEmbedder

if __name__ == "__main__":

    # 1. Create an ImageEmbedder with swav
    # Check out https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav
    for backbone in ["resnet50", "swav-imagenet"]:
        embedder = ImageEmbedder(backbone=backbone, embedding_dim=128)

        # 2. Generate an embedding from an image path.
        embeddings = embedder.predict('data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg')

        # 3. Assert dimension
        assert embeddings.shape == torch.Size((1, 128))

        # 4. Create a tensor random image
        random_img = torch.randn(1, 3, 32, 32)

        # 5. Generate an embedding from this random image
        embeddings = embedder.predict(random_img)

        # 6. Assert dimension
        assert embeddings.shape == torch.Size((1, 128))
