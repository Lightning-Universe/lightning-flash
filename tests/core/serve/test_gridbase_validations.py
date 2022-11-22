import pytest

from flash.core.serve import expose, ModelComponent
from flash.core.serve.types import Number
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _SERVE_TESTING


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_if_expose_decorator_not_applied_to_method():
    with pytest.raises(SyntaxError, match=r"expose.* decorator"):

        class FailedNoExposed(ModelComponent):
            def __init__(self, model):
                pass


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_if_more_than_one_expose_decorator_applied():
    with pytest.raises(SyntaxError, match=r"decorator must be applied to one"):

        class FailedTwoExposed(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def clasify(param):
                return param


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_if_first_arg_in_init_is_not_model():
    with pytest.raises(SyntaxError, match="__init__ must set 'model' as first"):

        class FailedModelArg(ModelComponent):
            def __init__(self, foo):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_if_second_arg_is_not_config():
    with pytest.raises(SyntaxError, match="__init__ can only set 'config'"):

        class FailedConfig(ModelComponent):
            def __init__(self, model, OTHER):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_if_random_parameters_in_init():
    with pytest.raises(SyntaxError, match="__init__ can only have 1 or 2 parameters"):

        class FailedInit(ModelComponent):
            def __init__(self, model, config, FOO):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_metaclass_raises_uses_restricted_method_name():
    # Restricted Name: `inputs`
    with pytest.raises(TypeError, match="bound methods/attrs named"):

        class FailedMethod_Inputs(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

            def inputs(self):
                pass

    # Restricted Name: `inputs`
    with pytest.raises(TypeError, match="bound methods/attrs named"):

        class FailedMethod_Outputs(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

            def outputs(self):
                pass

    # Restricted Name: `inputs`
    with pytest.raises(TypeError, match="bound methods/attrs named"):

        class FailedMethod_Name(ModelComponent):
            def __init__(self, model):
                pass

            @expose(inputs={"param": Number()}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

            @property
            def uid(self):
                return f"{self.uid}_SHOULD_NOT_RETURN"

    # Ensure that if we add more restricted names in the future,
    # there is a test for them as well.
    from flash.core.serve.component import _FLASH_SERVE_RESERVED_NAMES

    assert set(_FLASH_SERVE_RESERVED_NAMES).difference({"inputs", "outputs", "uid"}) == set()


def test_metaclass_raises_if_argument_values_of_expose_arent_subclasses_of_basetype():
    # try in `inputs` field
    with pytest.raises(TypeError, match="must be subclass of"):

        class FailedExposedDecoratorInputs(ModelComponent):
            def __init__(self, model):
                self.model = model

            @expose(inputs={"param": int}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param

    # try in `outputs` field
    with pytest.raises(TypeError, match="must be subclass of"):

        class FailedExposedDecoratorOutputs(ModelComponent):
            def __init__(self, model):
                self.model = model

            @expose(inputs={"param": Number()}, outputs={"foo": int})
            @staticmethod
            def predict(param):
                return param

    # try to pass a class definition, not an instance
    with pytest.raises(TypeError, match="must be subclass of"):

        class FailedExposedDecoratorClass(ModelComponent):
            def __init__(self, model):
                self.model = model

            @expose(inputs={"param": Number}, outputs={"foo": Number()})
            @staticmethod
            def predict(param):
                return param


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
def test_ModelComponent_raises_if_exposed_input_keys_differ_from_decorated_method_parameters(
    lightning_squeezenet1_1_obj,
):
    """This occurs when the instance is being initialized.

    This is noted because it differs from some other metaclass validations which will raise an exception at class
    definition time.
    """
    from tests.core.serve.models import ClassificationInference

    class FailedExposedDecorator(ModelComponent):
        def __init__(self, model):
            self.model = model

        @expose(inputs={"NOT_NAMED": Number()}, outputs={"foo": Number()})
        def predict(self, param):
            return param

    comp = ClassificationInference(lightning_squeezenet1_1_obj)

    with pytest.raises(RuntimeError, match="`@expose` must list all method arguments"):
        _ = FailedExposedDecorator(comp)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve is not installed.")
def test_ModelComponent_raises_if_config_is_empty_dict(lightning_squeezenet1_1_obj):
    """This occurs when the instance is being initialized.

    This is noted because it differs from some other metaclass validations which will raise an exception at class
    definition time.
    """

    class ConfigComponent(ModelComponent):
        def __init__(self, model, config):
            pass

        @expose(inputs={"param": Number()}, outputs={"foo": Number()})
        def predict(self, param):
            return param

    with pytest.raises(ValueError, match="dict of length < 1"):
        _ = ConfigComponent(lightning_squeezenet1_1_obj, config={})


@pytest.mark.skipif(not _CYTOOLZ_AVAILABLE, reason="the library cytoolz is not installed.")
def test_ModelComponent_raises_if_model_is_empty_iterable():
    """This occurs when the instance is being initialized.

    This is noted because it differs from some other metaclass validations which will raise an exception at class
    definition time.
    """

    class ConfigComponent(ModelComponent):
        def __init__(self, model):
            pass

        @expose(inputs={"param": Number()}, outputs={"foo": Number()})
        def predict(self, param):
            return param

    with pytest.raises(ValueError, match="must have length >= 1"):
        _ = ConfigComponent([])
