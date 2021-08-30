from flash.core.integrations.labelstudio.data_source import LabelStudioDataSource


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
    ds = LabelStudioDataSource._load_json_data(data=data, data_folder=".", multi_label=False)
    assert ds[3] == {"image"}
    assert ds[2] == {"Road", "Car", "Obstacle"}
    assert len(ds[1]) == 0
    assert len(ds[0]) == 5
    ds_multi = LabelStudioDataSource._load_json_data(data=data, data_folder=".", multi_label=True)
    assert ds_multi[3] == {"image"}
    assert ds_multi[2] == {"Road", "Car", "Obstacle"}
    assert len(ds_multi[1]) == 0
    assert len(ds_multi[0]) == 5
