from typing import Dict, Optional, Tuple

from flash.core.serve.component import ModelComponent
from flash.core.serve.core import Endpoint
from flash.core.serve.types import Repeated
from flash.core.utilities.imports import _PYDANTIC_AVAILABLE, _SERVE_TESTING

# Skip doctests if requirements aren't available
if not _SERVE_TESTING:
    __doctest_skip__ = ["EndpointProtocol.*"]

if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, create_model
else:
    BaseModel, create_model = object, None

try:
    from typing import ForwardRef

    RequestModel = ForwardRef("RequestModel")
    ResponseModel = ForwardRef("ResponseModel")
except ImportError:
    RequestModel = None
    ResponseModel = None


class Alive(BaseModel):
    """Represent the alive-result of the endpoint ``/alive``."""

    alive: bool  # skipcq: PTC-W0052


class EndpointProtocol:
    """Records the model classes used to define an endpoints request/response body.

    The request / response body schemas are generated dynamically depending on the endpoint + components passed into the
    class initializer. Component inputs & outputs (as defined in `@expose` object decorations) dtype method (`serialize`
    and `deserialize`) type hints are inspected in order to constuct a specification unique to the endpoint, they are
    returned as subclasses of pydantic ``BaseModel``.
    """

    def __init__(self, name: str, endpoint: "Endpoint", components: Dict[str, "ModelComponent"]):
        self._name = name
        self._endpoint = endpoint
        self._component = components

    @property
    def name(self) -> str:
        """Name assigned to the endpoint definition in the composition."""
        return self._name

    @property
    def route(self) -> str:
        """Endpoint HTTP route."""
        return self._endpoint.route

    @property
    def dsk_input_key_map(self) -> Dict[str, str]:
        """Map of payload key name -> key to insert in dsk before execution."""
        return self._endpoint.inputs

    @property
    def dsk_output_key_map(self):
        """Map output key names -> dsk output key names."""
        return self._endpoint.outputs

    @property
    def request_model(self) -> RequestModel:
        """Subclass of pydantic ``BaseModel`` specifying HTTP request body schema.

        Notes
        -----
        * Because pydantic does not allow you to define two models with
          the same `model name`, even when they are assigned to different
          python variables and contain different fields:

          >>> image_1 = create_model('Image', ...)  # doctest: +SKIP
          >>> image_2 = create_model('Image', ...)  # doctest: +SKIP
          >>> payload = create_model("Payload_1", **{"payload": image_1})  # doctest: +SKIP
          ERROR: Exception in ASGI application
          Traceback (most recent call last):
            ...
            model_name = model_name_map[model]
          KeyError: <class 'Image'>

          We prepend the name of the endpoint (which must be unique since
          endpoints are stored as a dict mapping names -> definitions within
          the composition) to the model class title. While this means that there
          are a lot of models defined within the OpenAPI scheam, this does not
          impact the field names of each models.

          As an examples: a model is created which will be a subfield of a
          "payload" model. The endpoint is named "classify_endpoint". The
          model we are defined will contains an encoded image string field.
          The model's name in the OpenAPI definition will be listed as
          "Classify_Endpoint_Image", but the field name "image" is untouched.
          Any POST to that endpoint just needs to send a json struct with
          the key "image" -> the raw data... The field names are NOT altered,
          and therefore this workaround should pose very little issue for
          our end users).
        """
        attrib_dict = {}
        inputs = self._endpoint.inputs
        for payload_name, component_and_input_key in inputs.items():
            component, _, key = component_and_input_key.split(".")
            param = self._component[component].inputs[key]
            hints = param.datatype.type_hints["input_args"]
            each = {}
            for key, key_t in hints.items():
                each[key] = (key_t, ...)
            model = create_model(f"{self.name.title()}_{payload_name.title()}", **each)
            if isinstance(param.datatype, Repeated):
                attrib_dict[payload_name] = (
                    Tuple[model, ...],
                    ...,
                )
            else:
                attrib_dict[payload_name] = (
                    model,
                    ...,
                )

        payload_model = create_model(f"{self.name.title()}_Payload", **attrib_dict)
        RequestModel = create_model(
            f"{self.name.title()}_RequestModel",
            __module__=self.__class__.__module__,
            **{"session": (Optional[str], None), "payload": (payload_model, ...)},
        )
        RequestModel.update_forward_refs()
        return RequestModel

    @property
    def response_model(self) -> ResponseModel:
        """Subclass of pydantic ``BaseModel`` specifying HTTP response body schema.

        Notes
        -----
        * Because pydantic does not allow you to define two models with
          the same `model name`, even when they are assigned to different
          python variables and contain different fields:

          >>> image_1 = create_model('Image', ...)  # doctest: +SKIP
          >>> image_2 = create_model('Image', ...)  # doctest: +SKIP
          >>> payload = create_model("Payload_1", **{"payload": image_1})  # doctest: +SKIP
          ERROR: Exception in ASGI application
          Traceback (most recent call last):
            ...
            model_name = model_name_map[model]
          KeyError: <class 'Image'>

          We prepend the name of the endpoint (which must be unique since
          endpoints are stored as a dict mapping names -> definitions within
          the composition) to the model class title. While this means that there
          are a lot of models defined within the OpenAPI scheam, this does not
          impact the field names of each models.

          As an examples: a model is created which will be a subfield of a
          "payload" model. The endpoint is named "classify_endpoint". The
          model we are defined will contains an encoded image string field.
          The model's name in the OpenAPI definition will be listed as
          "Classify_Endpoint_Image", but the field name "image" is untouched.
          Any POST to that endpoint just needs to send a json struct with
          the key "image" -> the raw data... The field names are NOT altered,
          and therefore this workaround should pose very little issue for
          our end users).
        """
        attrib_dict = {}
        outputs = self._endpoint.outputs
        for payload_name, component_and_output_key in outputs.items():
            component, _, key = component_and_output_key.split(".")
            param = self._component[component].outputs[key]
            hints = param.datatype.type_hints["output_args"]
            if isinstance(param.datatype, Repeated):
                attrib_dict[payload_name] = (
                    Tuple[hints, ...],
                    ...,
                )
            else:
                attrib_dict[payload_name] = (hints, ...)

        results_model = create_model(f"{self.name.title()}_Results", **attrib_dict)
        ResponseModel = create_model(
            f"{self.name.title()}_Response",
            __module__=self.__class__.__module__,
            **{"session": (Optional[str], None), "result": (results_model, ...)},
        )
        ResponseModel.update_forward_refs()
        return ResponseModel
