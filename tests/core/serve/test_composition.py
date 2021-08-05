import base64
from dataclasses import asdict

import pytest

from flash.core.serve import Composition, Endpoint
from flash.core.utilities.imports import _FASTAPI_AVAILABLE
from tests.helpers.utils import _SERVE_TESTING

if _FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_composit_endpoint_data(lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceComposable

    comp = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    composit = Composition(comp=comp)
    assert composit.component_uid_names == {"callnum_1": "comp"}
    assert composit.connections == []

    actual_endpoints = {k: asdict(v) for k, v in composit.endpoints.items()}
    assert actual_endpoints == {
        "classify_ENDPOINT": {
            "inputs": {"img": "callnum_1.inputs.img", "tag": "callnum_1.inputs.tag"},
            "outputs": {
                "cropped_img": "callnum_1.outputs.cropped_img",
                "predicted_tag": "callnum_1.outputs.predicted_tag",
            },
            "route": "/classify",
        }
    }

    ep = Endpoint(
        route="/predict",
        inputs={
            "label_1": comp.inputs.img,
            "tag_1": comp.inputs.tag,
        },
        outputs={
            "prediction": comp.outputs.predicted_tag,
            "cropped": comp.outputs.cropped_img,
        },
    )
    composit = Composition(comp=comp, predict_ep=ep)
    actual_endpoints = {k: asdict(v) for k, v in composit.endpoints.items()}
    assert actual_endpoints == {
        "predict_ep": {
            "inputs": {"label_1": "callnum_1.inputs.img", "tag_1": "callnum_1.inputs.tag"},
            "outputs": {
                "cropped": "callnum_1.outputs.cropped_img",
                "prediction": "callnum_1.outputs.predicted_tag",
            },
            "route": "/predict",
        }
    }


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_endpoint_errors_on_wrong_key_name(lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceComposable

    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    # input key does not exist
    with pytest.raises(AttributeError):
        _ = Endpoint(
            route="/predict",
            inputs={
                "label_1": comp1.inputs.img,
                "tag_1": comp1.inputs.DOESNOTEXIST,
            },
            outputs={
                "prediction": comp1.outputs.predicted_tag,
                "cropped": comp1.outputs.cropped_img,
            },
        )

    # output key does not exist
    with pytest.raises(AttributeError):
        _ = Endpoint(
            route="/predict",
            inputs={
                "label_1": comp1.inputs.img,
                "tag_1": comp1.inputs.tag,
            },
            outputs={
                "prediction": comp1.outputs.predicted_tag,
                "cropped": comp1.outputs.DOESNOTEXIST,
            },
        )

    # output key does not exist
    ep = Endpoint(
        route="/predict",
        inputs={
            "label_1": comp1.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={
            "prediction": comp1.outputs.predicted_tag,
            "cropped": "callnum_1.outputs.DOESNOTEXIST",
        },
    )
    with pytest.raises(AttributeError):
        _ = Composition(comp1=comp1, predict_ep=ep)

    # input function does not exist
    ep = Endpoint(
        route="/predict",
        inputs={
            "label_1": comp1.inputs.img,
            "tag_1": "DOESNOTEXIST.inputs.tag",
        },
        outputs={
            "prediction": comp1.outputs.predicted_tag,
            "cropped": comp1.outputs.cropped_img,
        },
    )
    with pytest.raises(AttributeError):
        _ = Composition(comp1=comp1, predict_ep=ep)

    # output function does not exist
    ep = Endpoint(
        route="/predict",
        inputs={
            "label_1": comp1.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={
            "prediction": comp1.outputs.predicted_tag,
            "cropped": "DOESNOTEXIST.outputs.cropped_img",
        },
    )
    with pytest.raises(AttributeError):
        _ = Composition(comp1=comp1, predict_ep=ep)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_composition_recieve_wrong_arg_type(lightning_squeezenet1_1_obj):
    # no endpoints or components
    with pytest.raises(TypeError):
        _ = Composition(hello="world")

    # no endpoints multiple components
    from tests.core.serve.models import ClassificationInferenceComposable

    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(ValueError):
        _ = Composition(c1=comp1, c2=comp2)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_servable_sequence(tmp_path, lightning_squeezenet1_1_obj, squeezenet_servable):
    from tests.core.serve.models import ClassificationInferenceModelSequence

    squeezenet_gm, _ = squeezenet_servable
    model_seq = [squeezenet_gm, squeezenet_gm]
    comp = ClassificationInferenceModelSequence(model_seq)

    composit = Composition(comp=comp)
    assert composit.components["callnum_1"]._flashserve_meta_.models == model_seq
    assert composit.components["callnum_1"].model1 == model_seq[0]
    assert composit.components["callnum_1"].model2 == model_seq[1]


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_servable_mapping(tmp_path, lightning_squeezenet1_1_obj, squeezenet_servable):
    from tests.core.serve.models import ClassificationInferenceModelMapping

    squeezenet_gm, _ = squeezenet_servable
    model_map = {"model_one": squeezenet_gm, "model_two": squeezenet_gm}
    comp = ClassificationInferenceModelMapping(model_map)

    composit = Composition(comp=comp)
    assert composit.components["callnum_1"]._flashserve_meta_.models == model_map
    assert composit.components["callnum_1"].model1 == model_map["model_one"]
    assert composit.components["callnum_1"].model2 == model_map["model_two"]


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_invalid_servable_composition(tmp_path, lightning_squeezenet1_1_obj, squeezenet_servable):
    from tests.core.serve.models import ClassificationInferenceModelMapping

    squeezenet_gm, _ = squeezenet_servable

    invalid_model_map = {"model_one": squeezenet_gm, "model_two": 235}
    with pytest.raises(TypeError):
        _ = ClassificationInferenceModelMapping(invalid_model_map)

    with pytest.raises(TypeError):
        _ = ClassificationInferenceModelMapping(lambda x: x + 1)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_complex_spec_single_endpoint(tmp_path, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceComposable

    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp3 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    comp1.outputs.predicted_tag >> comp3.inputs.tag  # skipcq: PYL-W0104
    comp2.outputs.cropped_img >> comp3.inputs.img  # skipcq: PYL-W0104
    comp1.outputs.predicted_tag >> comp2.inputs.tag  # skipcq: PYL-W0104

    ep = Endpoint(
        route="/predict",
        inputs={
            "img_1": comp1.inputs.img,
            "img_2": comp2.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={"prediction": comp3.outputs.predicted_tag},
    )

    composit = Composition(comp1=comp1, comp2=comp2, comp3=comp3, predict_compositon_ep=ep)
    connections = [str(c) for c in composit.connections]
    assert connections == [
        "callnum_1.outputs.predicted_tag >> callnum_3.inputs.tag",
        "callnum_1.outputs.predicted_tag >> callnum_2.inputs.tag",
        "callnum_2.outputs.cropped_img >> callnum_3.inputs.img",
    ]
    assert composit.component_uid_names == {
        "callnum_1": "comp1",
        "callnum_2": "comp2",
        "callnum_3": "comp3",
    }

    actual_endpoints = {k: asdict(v) for k, v in composit.endpoints.items()}
    assert actual_endpoints == {
        "predict_compositon_ep": {
            "inputs": {
                "img_1": "callnum_1.inputs.img",
                "img_2": "callnum_2.inputs.img",
                "tag_1": "callnum_1.inputs.tag",
            },
            "outputs": {
                "prediction": "callnum_3.outputs.predicted_tag",
            },
            "route": "/predict",
        }
    }


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_complex_spec_multiple_endpoints(tmp_path, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceComposable

    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp3 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    comp1.outputs.predicted_tag >> comp3.inputs.tag  # skipcq: PYL-W0104
    comp2.outputs.cropped_img >> comp3.inputs.img  # skipcq: PYL-W0104
    comp1.outputs.predicted_tag >> comp2.inputs.tag  # skipcq: PYL-W0104

    ep1 = Endpoint(
        route="/predict",
        inputs={
            "img_1": comp1.inputs.img,
            "img_2": comp2.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={"prediction": comp3.outputs.predicted_tag},
    )

    ep2 = Endpoint(
        route="/other_predict",
        inputs={
            "img_1": comp1.inputs.img,
            "img_2": comp2.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={
            "prediction_3": comp3.outputs.predicted_tag,
            "prediction_2": comp2.outputs.cropped_img,
        },
    )

    composit = Composition(comp1=comp1, comp2=comp2, comp3=comp3, predict_compositon_ep=ep1, other_predict_ep=ep2)
    connections = [str(c) for c in composit.connections]
    assert connections == [
        "callnum_1.outputs.predicted_tag >> callnum_3.inputs.tag",
        "callnum_1.outputs.predicted_tag >> callnum_2.inputs.tag",
        "callnum_2.outputs.cropped_img >> callnum_3.inputs.img",
    ]
    assert composit.component_uid_names == {
        "callnum_1": "comp1",
        "callnum_2": "comp2",
        "callnum_3": "comp3",
    }

    actual_endpoints = {k: asdict(v) for k, v in composit.endpoints.items()}
    assert actual_endpoints == {
        "predict_compositon_ep": {
            "inputs": {
                "img_1": "callnum_1.inputs.img",
                "img_2": "callnum_2.inputs.img",
                "tag_1": "callnum_1.inputs.tag",
            },
            "outputs": {
                "prediction": "callnum_3.outputs.predicted_tag",
            },
            "route": "/predict",
        },
        "other_predict_ep": {
            "inputs": {
                "img_1": "callnum_1.inputs.img",
                "img_2": "callnum_2.inputs.img",
                "tag_1": "callnum_1.inputs.tag",
            },
            "outputs": {
                "prediction_3": "callnum_3.outputs.predicted_tag",
                "prediction_2": "callnum_2.outputs.cropped_img",
            },
            "route": "/other_predict",
        },
    }


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_start_server_from_composition(tmp_path, squeezenet_servable, session_global_datadir):
    from tests.core.serve.models import ClassificationInferenceComposable

    squeezenet_gm, _ = squeezenet_servable
    comp1 = ClassificationInferenceComposable(squeezenet_gm)
    comp2 = ClassificationInferenceComposable(squeezenet_gm)
    comp3 = ClassificationInferenceComposable(squeezenet_gm)

    comp1.outputs.predicted_tag >> comp3.inputs.tag  # skipcq: PYL-W0104
    comp2.outputs.cropped_img >> comp3.inputs.img  # skipcq: PYL-W0104
    comp1.outputs.predicted_tag >> comp2.inputs.tag  # skipcq: PYL-W0104

    ep1 = Endpoint(
        route="/predict",
        inputs={
            "img_1": comp1.inputs.img,
            "img_2": comp2.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={"prediction": comp3.outputs.predicted_tag},
    )

    ep2 = Endpoint(
        route="/other_predict",
        inputs={
            "img_1": comp1.inputs.img,
            "img_2": comp2.inputs.img,
            "tag_1": comp1.inputs.tag,
        },
        outputs={
            "prediction_3": comp3.outputs.predicted_tag,
            "prediction_2": comp2.outputs.cropped_img,
        },
    )

    composit = Composition(
        comp1=comp1,
        comp2=comp2,
        comp3=comp3,
        predict_compositon_ep=ep1,
        other_predict_ep=ep2,
        TESTING=True,
        DEBUG=True,
    )

    with (session_global_datadir / "cat.jpg").open("rb") as f:
        cat_imgstr = base64.b64encode(f.read()).decode("UTF-8")
    with (session_global_datadir / "fish.jpg").open("rb") as f:
        fish_imgstr = base64.b64encode(f.read()).decode("UTF-8")
    data = {
        "session": "session_uuid",
        "payload": {
            "img_1": {"data": cat_imgstr},
            "img_2": {"data": fish_imgstr},
            "tag_1": {"label": "stingray"},
        },
    }
    expected_response = {
        "result": {"prediction": "goldfish, Carassius auratus"},
        "session": "session_uuid",
    }

    app = composit.serve(host="0.0.0.0", port=8000)
    with TestClient(app) as tc:
        res = tc.post("http://127.0.0.1:8000/predict", json=data)
        assert res.status_code == 200
        assert res.json() == expected_response
