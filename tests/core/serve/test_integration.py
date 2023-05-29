import base64

import pytest
from flash.core.serve import Composition, Endpoint
from flash.core.utilities.imports import _FASTAPI_AVAILABLE, _TOPIC_SERVE_AVAILABLE

if _TOPIC_SERVE_AVAILABLE:
    from jinja2 import TemplateNotFound
else:
    TemplateNotFound = ...

if _FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_resnet_18_inference_class(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInference

    comp = ClassificationInference(lightning_squeezenet1_1_obj)
    composit = Composition(comp=comp, TESTING=True, DEBUG=True)
    app = composit.serve(host="0.0.0.0", port=8000)

    with TestClient(app) as tc:
        alive = tc.get("http://127.0.0.1:8000/flashserve/alive")
        assert alive.status_code == 200
        assert alive.json() == {"alive": True}

        meta = tc.get("http://127.0.0.1:8000/classify/dag_json")
        assert isinstance(meta.json(), dict)

        meta = tc.get("http://127.0.0.1:8000/classify/meta")
        assert meta.status_code == 200

        with (session_global_datadir / "fish.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {"session": "UUID", "payload": {"img": {"data": imgstr}}}
        resp = tc.post("http://127.0.0.1:8000/classify", json=body)
        assert "result" in resp.json()
        expected = {"session": "UUID", "result": {"prediction": "goldfish, Carassius auratus"}}
        assert expected == resp.json()


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_start_server_with_repeated_exposed(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceRepeated

    comp = ClassificationInferenceRepeated(lightning_squeezenet1_1_obj)
    composit = Composition(comp=comp, TESTING=True, DEBUG=True)
    app = composit.serve(host="0.0.0.0", port=8000)
    with TestClient(app) as tc:
        meta = tc.get("http://127.0.0.1:8000/classify/meta")
        assert meta.status_code == 200
        with (session_global_datadir / "fish.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {"session": "UUID", "payload": {"img": [{"data": imgstr}]}}
        resp = tc.post("http://127.0.0.1:8000/classify", json=body)
        assert "result" in resp.json()
        expected = {
            "session": "UUID",
            "result": {
                "prediction": ["goldfish, Carassius auratus", "goldfish, Carassius auratus"],
                "other": 21,
            },
        }
        assert resp.json() == expected


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_serving_single_component_and_endpoint_no_composition(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInference

    comp = ClassificationInference(lightning_squeezenet1_1_obj)
    assert hasattr(comp.inputs, "img")
    assert hasattr(comp.outputs, "prediction")
    assert list(comp._flashserve_meta_.connections) == []

    ep = Endpoint(
        route="/different_route",
        inputs={"ep_in_image": comp.inputs.img},
        outputs={"ep_out_prediction": comp.outputs.prediction},
    )

    assert ep.route == "/different_route"

    composit = Composition(comp=comp, ep=ep, TESTING=True, DEBUG=True)
    app = composit.serve(host="0.0.0.0", port=8000)

    with TestClient(app) as tc:
        meta = tc.get("http://127.0.0.1:8000/different_route/meta")
        assert meta.json() == {
            "definitions": {
                "Ep_Ep_In_Image": {
                    "properties": {"data": {"title": "Data", "type": "string"}},
                    "required": ["data"],
                    "title": "Ep_Ep_In_Image",
                    "type": "object",
                },
                "Ep_Payload": {
                    "properties": {"ep_in_image": {"$ref": "#/definitions/Ep_Ep_In_Image"}},
                    "required": ["ep_in_image"],
                    "title": "Ep_Payload",
                    "type": "object",
                },
            },
            "properties": {
                "payload": {"$ref": "#/definitions/Ep_Payload"},
                "session": {"title": "Session", "type": "string"},
            },
            "required": ["payload"],
            "title": "Ep_RequestModel",
            "type": "object",
        }

        with (session_global_datadir / "fish.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {"session": "UUID", "payload": {"ep_in_image": {"data": imgstr}}}
        success = tc.post("http://127.0.0.1:8000/different_route", json=body)
        assert tc.post("http://127.0.0.1:8000/classify", json=body).status_code == 404
        assert tc.post("http://127.0.0.1:8000/my_test_component", json=body).status_code == 404

        assert "result" in success.json()
        expected = {
            "session": "UUID",
            "result": {"ep_out_prediction": "goldfish, Carassius auratus"},
        }
        assert expected == success.json()

        res = tc.get("http://127.0.0.1:8000/flashserve/dag_json")
        assert res.status_code == 200
        assert res.json() == {
            "component_dependencies": {
                "callnum_1": {
                    "callnum_1.funcout": ["callnum_1.inputs.img"],
                    "callnum_1.inputs.img": [],
                    "callnum_1.outputs.prediction": ["callnum_1.funcout"],
                    "callnum_1.outputs.prediction.serial": ["callnum_1.outputs.prediction"],
                }
            },
            "component_dependents": {
                "callnum_1": {
                    "callnum_1.funcout": ["callnum_1.outputs.prediction"],
                    "callnum_1.inputs.img": ["callnum_1.funcout"],
                    "callnum_1.outputs.prediction": ["callnum_1.outputs.prediction.serial"],
                    "callnum_1.outputs.prediction.serial": [],
                }
            },
            "component_funcnames": {
                "callnum_1": {
                    "callnum_1.funcout": ["Compose"],
                    "callnum_1.inputs.img": ["packed_deserialize"],
                    "callnum_1.outputs.prediction": ["get"],
                    "callnum_1.outputs.prediction.serial": ["serialize"],
                }
            },
            "connections": [],
        }


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
@pytest.mark.xfail(TemplateNotFound, reason="jinja2.exceptions.TemplateNotFound: dag.html")  # todo
def test_serving_composed(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInference, SeatClassifier

    resnet_comp = ClassificationInference(lightning_squeezenet1_1_obj)
    seat_comp = SeatClassifier(lightning_squeezenet1_1_obj, config={"sport": "football"})
    resnet_comp.outputs.prediction >> seat_comp.inputs.stadium
    ep = Endpoint(
        route="/predict_seat",
        inputs={
            "image": resnet_comp.inputs.img,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={
            "seat_number": seat_comp.outputs.seat_number,
            "team": seat_comp.outputs.team,
        },
    )
    composit = Composition(
        resnet_comp=resnet_comp,
        seat_comp=seat_comp,
        predict_seat_ep=ep,
        TESTING=True,
        DEBUG=True,
    )
    app = composit.serve(host="0.0.0.0", port=8000)

    with TestClient(app) as tc:
        meta = tc.get("http://127.0.0.1:8000/predict_seat/meta")
        assert meta.status_code == 200

        with (session_global_datadir / "cat.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {
            "session": "UUID",
            "payload": {
                "image": {"data": imgstr},
                "section": {"num": 10},
                "isle": {"num": 4},
                "row": {"num": 53},
            },
        }
        success = tc.post("http://127.0.0.1:8000/predict_seat", json=body)
        assert success.json() == {
            "result": {"seat_number": 4799680, "team": "buffalo bills, the ralph"},
            "session": "UUID",
        }
        resp = tc.get("http://127.0.0.1:8000/predict_seat/dag")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
@pytest.mark.xfail(TemplateNotFound, reason="jinja2.exceptions.TemplateNotFound: dag.html")  # todo
def test_composed_does_not_eliminate_endpoint_serialization(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInference, SeatClassifier

    resnet_comp = ClassificationInference(lightning_squeezenet1_1_obj)
    seat_comp = SeatClassifier(lightning_squeezenet1_1_obj, config={"sport": "football"})

    resnet_comp.outputs.prediction >> seat_comp.inputs.stadium

    ep = Endpoint(
        route="/predict_seat",
        inputs={
            "image": resnet_comp.inputs.img,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={
            "seat_number_out": seat_comp.outputs.seat_number,
            "team_out": seat_comp.outputs.team,
        },
    )
    ep2 = Endpoint(
        route="/predict_seat_img",
        inputs={
            "image": resnet_comp.inputs.img,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={
            "seat_number_out": seat_comp.outputs.seat_number,
            "team_out": seat_comp.outputs.team,
            "image_out": resnet_comp.outputs.prediction,
        },
    )

    composit = Composition(
        resnet_comp=resnet_comp,
        seat_comp=seat_comp,
        seat_prediction_ep=ep,
        seat_image_prediction_ep=ep2,
        TESTING=True,
        DEBUG=True,
    )
    app = composit.serve(host="0.0.0.0", port=8000)

    with TestClient(app) as tc:
        meta = tc.get("http://127.0.0.1:8000/predict_seat/meta")
        assert meta.status_code == 200

        meta = tc.get("http://127.0.0.1:8000/predict_seat_img/meta")
        assert meta.status_code == 200

        with (session_global_datadir / "cat.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {
            "session": "UUID",
            "payload": {
                "image": {"data": imgstr},
                "section": {"num": 10},
                "isle": {"num": 4},
                "row": {"num": 53},
            },
        }
        success = tc.post("http://127.0.0.1:8000/predict_seat", json=body)
        assert success.json() == {
            "result": {"seat_number_out": 4799680, "team_out": "buffalo bills, the ralph"},
            "session": "UUID",
        }
        resp = tc.get("http://127.0.0.1:8000/predict_seat/dag")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
@pytest.mark.xfail(TemplateNotFound, reason="jinja2.exceptions.TemplateNotFound: dag.html")  # todo
def test_endpoint_overwrite_connection_dag(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInference, SeatClassifier

    resnet_comp = ClassificationInference(lightning_squeezenet1_1_obj)
    seat_comp = SeatClassifier(lightning_squeezenet1_1_obj, config={"sport": "football"})

    resnet_comp.outputs.prediction >> seat_comp.inputs.stadium

    ep = Endpoint(
        route="/predict_seat",
        inputs={
            "image": resnet_comp.inputs.img,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={"seat_number": seat_comp.outputs.seat_number, "team": seat_comp.outputs.team},
    )
    ep2 = Endpoint(
        route="/predict_seat_img",
        inputs={
            "image": resnet_comp.inputs.img,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={
            "seat_number": seat_comp.outputs.seat_number,
            "team": seat_comp.outputs.team,
            "img_out": resnet_comp.outputs.prediction,
        },
    )
    ep3 = Endpoint(
        route="/predict_seat_img_two",
        inputs={
            "stadium": seat_comp.inputs.stadium,
            "isle": seat_comp.inputs.isle,
            "section": seat_comp.inputs.section,
            "row": seat_comp.inputs.row,
        },
        outputs={"seat_number": seat_comp.outputs.seat_number, "team": seat_comp.outputs.team},
    )

    composit = Composition(
        resnet_comp=resnet_comp,
        seat_comp=seat_comp,
        seat_prediction_ep=ep,
        seat_image_prediction_ep=ep2,
        seat_image_prediction_two_ep=ep3,
        TESTING=True,
        DEBUG=True,
    )
    app = composit.serve(host="0.0.0.0", port=8000)

    with TestClient(app) as tc:
        resp = tc.get("http://127.0.0.1:8000/flashserve/component_dags")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"
        resp = tc.get("http://127.0.0.1:8000/predict_seat/dag")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"
        resp = tc.get("http://127.0.0.1:8000/predict_seat_img/dag")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"
        resp = tc.get("http://127.0.0.1:8000/predict_seat_img_two/dag")
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.template.name == "dag.html"

        with (session_global_datadir / "cat.jpg").open("rb") as f:
            imgstr = base64.b64encode(f.read()).decode("UTF-8")
        body = {
            "session": "UUID",
            "payload": {
                "image": {"data": imgstr},
                "section": {"num": 10},
                "isle": {"num": 4},
                "row": {"num": 53},
            },
        }
        success = tc.post("http://127.0.0.1:8000/predict_seat", json=body)
        assert success.json() == {
            "result": {"seat_number": 4799680, "team": "buffalo bills, the ralph"},
            "session": "UUID",
        }

        success = tc.post("http://127.0.0.1:8000/predict_seat_img", json=body)
        assert success.json() == {
            "result": {
                "seat_number": 4799680,
                "team": "buffalo bills, the ralph",
                "img_out": "Persian cat",
            },
            "session": "UUID",
        }

        body = {
            "session": "UUID",
            "payload": {
                "stadium": {"label": "buffalo bills, the ralph"},
                "section": {"num": 10},
                "isle": {"num": 4},
                "row": {"num": 53},
            },
        }
        success = tc.post("http://127.0.0.1:8000/predict_seat_img_two", json=body)
        assert success.json() == {
            "result": {"seat_number": 16960000, "team": "buffalo bills, the ralph"},
            "session": "UUID",
        }


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_cycle_in_connection_fails(session_global_datadir, lightning_squeezenet1_1_obj):
    from tests.core.serve.models import ClassificationInferenceComposable

    c1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(RuntimeError):
        c1.outputs.cropped_img >> c1.inputs.img


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_composition_from_url_torchscript_servable(tmp_path):
    from flash.core.serve import ModelComponent, Servable, expose
    from flash.core.serve.types import Number

    """
    # Tensor x Tensor
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, a, b):
            result_0 = a / b
            result_1 = torch.div(a, b)
            result_2 = a.div(b)

            return result_0, result_1, result_2

    TorchScript (.pt) can be downloaded at TORCHSCRIPT_DOWNLOAD_URL
    """
    TORCHSCRIPT_DOWNLOAD_URL = "https://github.com/pytorch/pytorch/raw/95489b590f00801bdee7f41783f30874883cf6bb/test/jit/fixtures/test_versioned_div_tensor_inplace_v3.pt"  # noqa E501

    class ComponentTwoModels(ModelComponent):
        def __init__(self, model):
            self.encoder = model["encoder"]
            self.decoder = model["decoder"]

        @expose(inputs={"inp": Number()}, outputs={"output": Number()})
        def do_my_predict(self, inp):
            """My predict docstring."""
            return self.decoder(self.encoder(inp, inp), inp)

    gm = Servable(TORCHSCRIPT_DOWNLOAD_URL, download_path=tmp_path / "tmp_download.pt")

    c_1 = ComponentTwoModels({"encoder": gm, "decoder": gm})
    c_2 = ComponentTwoModels({"encoder": gm, "decoder": gm})

    c_1.outputs.output >> c_2.inputs.inp

    ep = Endpoint(
        route="/predictr",
        inputs={"ep_in": c_1.inputs.inp},
        outputs={"ep_out": c_1.outputs.output},
    )

    composit = Composition(c_1=c_1, c_2=c_2, endpoints=ep, TESTING=True, DEBUG=True)
    app = composit.serve(host="0.0.0.0", port=8000)
    with TestClient(app) as tc:
        body = {
            "session": "UUID",
            "payload": {
                "ep_in": {"num": 10},
            },
        }
        success = tc.post("http://127.0.0.1:8000/predictr", json=body)
        assert success.json() == {
            "result": {"ep_out": 1.0},
            "session": "UUID",
        }
