.. beta:: The VISSL integration is currently in Beta.

.. _vissl:

#####
VISSL
#####

`VISSL <https://github.com/facebookresearch/vissl>`__ is a library from Facebook AI Research for state-of-the-art self-supervised learning.
We integrate VISSL models and algorithms into Flash with the :ref:`image embedder <image_embedder>` task.

Using VISSL with Flash
----------------------

The ImageEmbedder task in Flash can be configured with different backbones, projection heads, image transforms and loss functions so that you can train your feature extractor using a SOTA SSL method.

.. code-block:: python

    from flash.image import ImageEmbedder

    embedder = ImageEmbedder(
        backbone="resnet",
        training_strategy="barlow_twins",
        head="simclr_head",
        pretraining_transform="barlow_twins_transform",
        training_strategy_kwargs={"latent_embedding_dim": 256, "dims": [2048, 2048, 256]},
        pretraining_transform_kwargs={"size_crops": [196]},
    )

The user can pass arguments to the training strategy, image transforms and backbones using the optional dictionary arguments the ImageEmbedder task accepts.
The training strategies club together the projection head, the loss function as well as VISSL hooks for a particular algorithm and the arguments to customize these can passed via ``training_strategy_kwargs``.
As an example, in the above code block, the ``latent_embedding_dim`` is an argument to the BarlowTwins loss function from VISSL, while the ``dims`` argument configures the projection head to output 256 dim vectors for the loss function.

If you find VISSL integration in Flash useful for your research, please don't forget to cite us and the VISSL library.
You can find our bibtex on `Flash <https://github.com/PyTorchLightning/lightning-flash>`__ and VISSL's bibxtex on their `github <https://github.com/facebookresearch/vissl>`__ page.
