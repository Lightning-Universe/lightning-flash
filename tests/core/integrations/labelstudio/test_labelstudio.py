import pytest

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.utils import download_data
from flash.core.integrations.labelstudio.input import (
    _load_json_data,
    LabelStudioImageClassificationInput,
    LabelStudioInput,
    LabelStudioTextClassificationInput,
)
from flash.core.integrations.labelstudio.visualizer import launch_app
from flash.core.utilities.stages import RunningStage
from flash.image.classification.data import ImageClassificationData
from flash.text.classification.data import TextClassificationData
from flash.text.classification.model import TextClassificationBackboneState
from flash.video.classification.data import LabelStudioVideoClassificationInput, VideoClassificationData
from tests.helpers.utils import _IMAGE_TESTING, _TEXT_TESTING, _VIDEO_TESTING


def test_utility_load():
    """Test for label studio json loader."""
    data = [
        {
            "id": 191,
            "annotations": [
                {
                    "id": 130,
                    "completed_by": {"id": 1, "email": "test@heartex.com", "first_name": "", "last_name": ""},
                    "result": [
                        {
                            "id": "dv1Tn-zdez",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 46.5625,
                                "y": 21.666666666666668,
                                "width": 8.75,
                                "height": 12.083333333333334,
                                "rotation": 0,
                                "rectanglelabels": ["Car"],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "image_rotation": 0,
                            "original_width": 320,
                            "original_height": 240,
                        },
                        {
                            "id": "KRa8jEvpK0",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 66.875,
                                "y": 22.5,
                                "width": 14.0625,
                                "height": 17.5,
                                "rotation": 0,
                                "rectanglelabels": ["Car"],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "image_rotation": 0,
                            "original_width": 320,
                            "original_height": 240,
                        },
                        {
                            "id": "kAKaSxNnvH",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 93.4375,
                                "y": 22.916666666666668,
                                "width": 6.5625,
                                "height": 18.75,
                                "rotation": 0,
                                "rectanglelabels": ["Car"],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "image_rotation": 0,
                            "original_width": 320,
                            "original_height": 240,
                        },
                        {
                            "id": "_VXKV2nz14",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 0,
                                "y": 39.583333333333336,
                                "width": 100,
                                "height": 60.416666666666664,
                                "rotation": 0,
                                "rectanglelabels": ["Road"],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "image_rotation": 0,
                            "original_width": 320,
                            "original_height": 240,
                        },
                        {
                            "id": "vCuvi_jLHn",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 0,
                                "y": 17.5,
                                "width": 48.125,
                                "height": 41.66666666666666,
                                "rotation": 0,
                                "rectanglelabels": ["Obstacle"],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "image_rotation": 0,
                            "original_width": 320,
                            "original_height": 240,
                        },
                    ],
                    "was_cancelled": False,
                    "ground_truth": False,
                    "prediction": {},
                    "result_count": 0,
                    "task": 191,
                }
            ],
            "file_upload": "Highway20030201_1002591.jpg",
            "data": {"image": "/data/upload/Highway20030201_1002591.jpg"},
            "meta": {},
            "created_at": "2021-05-12T18:43:41.241095Z",
            "updated_at": "2021-05-12T19:42:28.156609Z",
            "project": 7,
        }
    ]
    ds = _load_json_data(data=data, data_folder=".", multi_label=False)
    assert ds[3] == {"image"}
    assert ds[2] == {"Road", "Car", "Obstacle"}
    assert len(ds[1]) == 0
    assert len(ds[0]) == 5
    ds_multi = _load_json_data(data=data, data_folder=".", multi_label=True)
    assert ds_multi[3] == {"image"}
    assert ds_multi[2] == {"Road", "Car", "Obstacle"}
    assert len(ds_multi[1]) == 0
    assert len(ds_multi[0]) == 5


def test_input_labelstudio():
    """Test creation of LabelStudioInput."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/data.zip")
    data = {
        "data_folder": "data/upload/",
        "export_json": "data/project.json",
        "multi_label": False,
    }

    data_pipeline_state = DataPipelineState()
    train_data, val_data = LabelStudioInput._split_train_val_data(data, split=0.1)
    train = LabelStudioInput(RunningStage.TRAINING, train_data, data_pipeline_state=data_pipeline_state)
    val = LabelStudioInput(RunningStage.VALIDATING, val_data, data_pipeline_state=data_pipeline_state)

    train_sample = train[0]
    val_sample = val[0]
    assert train_sample
    assert val_sample


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_input_labelstudio_image():
    """Test creation of LabelStudioImageClassificationInput from images."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/data_nofile.zip")

    data = {
        "data_folder": "data/upload/",
        "export_json": "data/project_nofile.json",
        "multi_label": True,
    }

    data_pipeline_state = DataPipelineState()
    train_data, val_data = LabelStudioInput._split_train_val_data(data, split=0.2)
    train = LabelStudioImageClassificationInput(
        RunningStage.TRAINING, train_data, data_pipeline_state=data_pipeline_state
    )
    val = LabelStudioImageClassificationInput(
        RunningStage.VALIDATING, val_data, data_pipeline_state=data_pipeline_state
    )
    assert train._data_pipeline_state == val._data_pipeline_state == data_pipeline_state

    train_sample = train[0]
    val_sample = val[0]
    assert train_sample
    assert val_sample


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_datamodule_labelstudio_image():
    """Test creation of LabelStudioImageClassificationInput and Datamodule from images."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/data.zip")

    datamodule = ImageClassificationData.from_labelstudio(
        train_export_json="data/project.json",
        train_data_folder="data/upload/",
        test_export_json="data/project.json",
        test_data_folder="data/upload/",
        val_split=0.5,
    )
    assert datamodule


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_label_studio_predictions_visualization():
    """Test creation of LabelStudioImageClassificationInput and Datamodule from images."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/data.zip")

    datamodule = ImageClassificationData.from_labelstudio(
        train_export_json="data/project.json",
        train_data_folder="data/upload/",
        test_export_json="data/project.json",
        test_data_folder="data/upload/",
        val_split=0.5,
    )
    assert datamodule
    app = launch_app(datamodule)
    predictions = [[0, 1], [1, 0]]
    vis_predictions = app.show_predictions(predictions)
    assert len(vis_predictions) == 4
    assert vis_predictions[0]["result"][0]["id"] != vis_predictions[3]["result"][0]["id"]
    assert vis_predictions[1]["result"][0]["id"] != vis_predictions[2]["result"][0]["id"]
    tasks_predictions = app.show_tasks(predictions)
    assert len(tasks_predictions) == 4
    tasks_predictions_json = app.show_tasks(predictions, export_json="data/project.json")
    assert tasks_predictions_json


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_input_labelstudio_text():
    """Test creation of LabelStudioTextClassificationInput and Datamodule from text."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/text_data.zip", "./data/")
    data = {
        "data_folder": "data/upload/",
        "export_json": "data/project.json",
        "multi_label": False,
    }

    data_pipeline_state = DataPipelineState()
    train_data, test_data = LabelStudioInput._split_train_test_data(data)
    train_data, val_data = LabelStudioInput._split_train_val_data(train_data, split=0.2)
    train = LabelStudioTextClassificationInput(
        RunningStage.TRAINING, train_data, data_pipeline_state=data_pipeline_state
    )
    val = LabelStudioTextClassificationInput(RunningStage.VALIDATING, val_data, data_pipeline_state=data_pipeline_state)
    test = LabelStudioTextClassificationInput(RunningStage.TESTING, test_data, data_pipeline_state=data_pipeline_state)

    backbone = "prajjwal1/bert-tiny"
    train.set_state(TextClassificationBackboneState(backbone))

    assert train._data_pipeline_state == val._data_pipeline_state
    assert train._data_pipeline_state == test._data_pipeline_state
    train_sample = train[0]
    val_sample = val[0]
    assert train_sample
    assert val_sample
    assert len(test) == 0


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_datamodule_labelstudio_text():
    """Test creation of LabelStudioTextClassificationInput and Datamodule from text."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/text_data.zip", "./data/")
    datamodule = TextClassificationData.from_labelstudio(
        train_export_json="data/project.json",
        data_folder="data/upload/",
    )
    assert datamodule


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_input_labelstudio_video():
    """Test creation of LabelStudioVideoClassificationInput from video."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/video_data.zip")
    data = {"data_folder": "data/upload/", "export_json": "data/project.json", "multi_label": True}

    train = LabelStudioVideoClassificationInput(RunningStage.TRAINING, data)
    assert train.data._labeled_videos[0] == ("data/upload/INO_Crossroads.avi", {"label": 0})
    sample_iter = iter(train)
    sample = next(sample_iter)
    assert train
    assert sample


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_datamodule_labelstudio_video():
    """Test creation of Datamodule from video."""
    download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/video_data.zip")
    datamodule = VideoClassificationData.from_labelstudio(
        export_json="data/project.json",
        data_folder="data/upload/",
        clip_sampler="uniform",
        clip_duration=1,
        decode_audio=False,
    )
    assert datamodule
