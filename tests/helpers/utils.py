import os

_IMAGE_TESTING = False
_VIDEO_TESTING = False
_TABULAR_TESTING = False
_TEXT_TESTING = False
_IMAGE_STLYE_TRANSFER_TESTING = False
_SERVE_TESTING = False

if "FLASH_TEST_TOPIC" in os.environ:
    topic = os.environ["FLASH_TEST_TOPIC"]
    _IMAGE_TESTING = topic == "image"
    _VIDEO_TESTING = topic == "video"
    _TABULAR_TESTING = topic == "tabular"
    _TEXT_TESTING = topic == "text"
    _IMAGE_STLYE_TRANSFER_TESTING = topic == "image_style_transfer"
    _SERVE_TESTING = topic == "serve"
