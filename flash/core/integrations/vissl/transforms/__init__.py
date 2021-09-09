from flash.core.utilities.imports import _VISSL_AVAILABLE  # noqa: F401

if _VISSL_AVAILABLE:
    from classy_vision.dataset.transforms import register_transform  # noqa: F401

    from flash.core.integrations.vissl.transforms.multicrop import StandardMultiCropSSLTransform  # noqa: F401
    from flash.core.integrations.vissl.transforms.utilities import vissl_collate_fn  # noqa: F401

    register_transform("multicrop_ssl_transform")(StandardMultiCropSSLTransform)
