from flash.vision import SemanticSegmentationData


def test_smoke():
    dm = SemanticSegmentationData()
    assert dm is not None
