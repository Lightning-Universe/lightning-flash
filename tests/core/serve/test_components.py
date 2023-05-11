import pytest
import torch

from flash.core.serve.types import Label
from flash.core.utilities.imports import _TOPIC_SERVE_AVAILABLE
from tests.core.serve.models import ClassificationInferenceComposable, LightningSqueezenet


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_model_compute_call_method(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    img = torch.arange(195075).reshape((1, 255, 255, 3))
    tag = None
    out_res, out_img = comp1(img, tag)
    assert out_res.item() == 753


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_model_compute_dependencies(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    comp1.inputs.tag << comp2.outputs.predicted_tag
    res = [
        {
            "source_component": "callnum_2",
            "source_key": "predicted_tag",
            "target_component": "callnum_1",
            "target_key": "tag",
        }
    ]
    assert [x._asdict() for x in comp1._flashserve_meta_.connections] == res
    assert list(comp2._flashserve_meta_.connections) == []


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_inverse_model_compute_component_dependencies(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    comp2.outputs.predicted_tag >> comp1.inputs.tag

    res = [
        {
            "source_component": "callnum_2",
            "source_key": "predicted_tag",
            "target_component": "callnum_1",
            "target_key": "tag",
        }
    ]
    assert [x._asdict() for x in comp2._flashserve_meta_.connections] == res
    assert list(comp1._flashserve_meta_.connections) == []


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_two_component_invalid_dependencies_fail(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(RuntimeError, match="Cannot create cycle"):
        comp1.inputs["tag"] << comp1.outputs.predicted_tag
    with pytest.raises(RuntimeError, match="Cannot create cycle"):
        comp1.inputs.tag << comp1.outputs["predicted_tag"]

    with pytest.raises(AttributeError):
        comp1.inputs["tag"] >> comp2.inputs["label"]
    with pytest.raises(AttributeError):
        comp1.inputs.tag >> comp2.inputs.label

    with pytest.raises(AttributeError):
        comp1.inputs["tag"] >> comp2.outputs["label"]
    with pytest.raises(AttributeError):
        comp1.inputs.tag >> comp2.outputs.label

    with pytest.raises(TypeError):
        comp2.outputs["predicted_tag"] >> comp1.outputs["predicted_tag"]
    with pytest.raises(TypeError):
        comp2.outputs.predicted_tag >> comp1.outputs.predicted_tag

    class Foo:
        def __init__(self):
            pass

    foo = Foo()
    with pytest.raises(TypeError):
        comp1.inputs["tag"] >> foo


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_component_initialization(lightning_squeezenet1_1_obj):
    with pytest.raises(TypeError):
        ClassificationInferenceComposable(wrongname=lightning_squeezenet1_1_obj)

    comp = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    assert comp.uid == "callnum_1"
    assert hasattr(comp.inputs, "img")
    assert hasattr(comp.inputs, "tag")
    assert hasattr(comp.outputs, "predicted_tag")
    assert hasattr(comp.outputs, "cropped_img")
    assert "img" in comp.inputs
    assert "predicted_tag" in comp.outputs


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_component_parameters(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)
    comp2 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(TypeError):
        # Immutability test
        comp1.inputs["newkey"] = comp2.inputs["tag"]

    first_tag = comp1.outputs["predicted_tag"]
    second_tag = comp2.inputs["tag"]
    assert isinstance(first_tag.datatype, Label)

    assert first_tag.connections == []
    first_tag >> second_tag
    assert str(first_tag.connections[0]) == ("callnum_1.outputs.predicted_tag >> callnum_2.inputs.tag")
    assert second_tag.connections == []
    assert first_tag.connections == comp1._flashserve_meta_.connections


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_invalid_expose_inputs():
    from flash.core.serve import ModelComponent, expose
    from flash.core.serve.types import Number

    lr = LightningSqueezenet()

    with pytest.raises(SyntaxError, match="must be valid python attribute"):

        class ComposeClassInvalidExposeNameKeyword(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={"def": Number()})
            @staticmethod
            def predict(param):
                return param

        _ = ComposeClassInvalidExposeNameKeyword(lr)

    with pytest.raises(AttributeError, match="object has no attribute"):

        class ComposeClassInvalidExposeNameType(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={12: Number()})
            @staticmethod
            def predict(param):
                return param

        _ = ComposeClassInvalidExposeNameType(lr)

    with pytest.raises(TypeError, match="`expose` values must be"):

        class ComposeClassInvalidExposeInputsType(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs=Number(), outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

        _ = ComposeClassInvalidExposeInputsType(lr)

    with pytest.raises(ValueError, match="cannot set dict of length < 1"):

        class ComposeClassEmptyExposeInputsType(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

        _ = ComposeClassEmptyExposeInputsType(lr)


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_connection_invalid_raises(lightning_squeezenet1_1_obj):
    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(RuntimeError, match="Cannot compose a parameters of same components"):
        comp1.outputs["predicted_tag"] >> comp1.outputs["predicted_tag"]

    class FakeParam:
        position = "outputs"

    fake_param = FakeParam()

    with pytest.raises(TypeError, match="Can only Compose another `Parameter`"):
        comp1.outputs.predicted_tag >> fake_param


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_invalid_name(lightning_squeezenet1_1_obj):
    from flash.core.serve import ModelComponent, expose
    from flash.core.serve.types import Number

    with pytest.raises(SyntaxError):

        class FailedExposedOutputsKeyworkName(ModelComponent):
            def __init__(self, model):
                self.model = model

            @expose(inputs={"param": Number()}, outputs={"def": Number()})
            @staticmethod
            def predict(param):
                return param


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_invalid_config_args(lightning_squeezenet1_1_obj):
    from flash.core.serve import ModelComponent, expose
    from flash.core.serve.types import Number

    class SomeComponent(ModelComponent):
        def __init__(self, model, config=None):
            self.model = model
            self.config = config

        @expose(inputs={"param": Number()}, outputs={"out": Number()})
        def predict(self, param):
            return param

    # not a dict
    with pytest.raises(TypeError, match="Config must be"):
        _ = SomeComponent(lightning_squeezenet1_1_obj, config="invalid")

    # not a str key
    with pytest.raises(TypeError, match="config key"):
        _ = SomeComponent(lightning_squeezenet1_1_obj, config={12: "value"})

    # not a primitive value
    with pytest.raises(TypeError, match="config val"):
        _ = SomeComponent(lightning_squeezenet1_1_obj, config={"key": lambda x: x})


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_invalid_model_args(lightning_squeezenet1_1_obj):
    from flash.core.serve import ModelComponent, expose
    from flash.core.serve.types import Number

    class SomeComponent(ModelComponent):
        def __init__(self, model):
            self.model = model

        @expose(inputs={"param": Number()}, outputs={"out": Number()})
        @staticmethod
        def predict(param):
            return param

    # not a valid object type
    with pytest.raises(TypeError):
        _ = SomeComponent("INVALID")

    # not a valid sequence
    with pytest.raises(TypeError):
        _ = SomeComponent([lightning_squeezenet1_1_obj, "invalid"])

    # not a valid key
    with pytest.raises(TypeError):
        _ = SomeComponent({"first": lightning_squeezenet1_1_obj, 23: lightning_squeezenet1_1_obj})

    # not a valid value
    with pytest.raises(TypeError):
        _ = SomeComponent({"first": lightning_squeezenet1_1_obj, "second": 233})


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
def test_create_invalid_endpoint(lightning_squeezenet1_1_obj):
    from flash.core.serve import Endpoint

    comp1 = ClassificationInferenceComposable(lightning_squeezenet1_1_obj)

    with pytest.raises(TypeError, match="route parameter must be type"):
        _ = Endpoint(
            route=b"/INVALID",
            inputs={"inp": comp1.inputs.img},
            outputs={"out": comp1.outputs.cropped_img},
        )

    with pytest.raises(ValueError, match="route must begin with"):
        _ = Endpoint(
            route="hello",
            inputs={"inp": comp1.inputs.img},
            outputs={"out": comp1.outputs.cropped_img},
        )

    with pytest.raises(TypeError, match="inputs k=inp, v=b'INVALID'"):
        _ = Endpoint(
            route="/hello",
            inputs={"inp": b"INVALID"},
            outputs={"out": comp1.outputs.cropped_img},
        )

    with pytest.raises(TypeError, match="k=out, v=b'INVALID'"):
        _ = Endpoint(route="/hello", inputs={"inp": comp1.inputs.img}, outputs={"out": b"INVALID"})
