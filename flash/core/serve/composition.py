import itertools
from dataclasses import asdict
from typing import Dict, List, Tuple, Union

from flash.core.serve.component import ModelComponent
from flash.core.serve.core import Connection, Endpoint
from flash.core.serve.interfaces.models import EndpointProtocol
from flash.core.serve.server import ServerMixin
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE

if _CYTOOLZ_AVAILABLE:
    from cytoolz import concat, first
else:
    concat, first = None, None


def _parse_composition_kwargs(
    **kwargs: Union[ModelComponent, Endpoint]
) -> Tuple[Dict[str, ModelComponent], Dict[str, Endpoint]]:

    components, endpoints = {}, {}
    for k, v in kwargs.items():
        if isinstance(v, ModelComponent):
            components[k] = v
        elif isinstance(v, Endpoint):
            endpoints[k] = v
        else:
            raise TypeError(f"{k}={v} is not valid type (recieved {type(v)}")

    if len(components) > 1 and len(endpoints) == 0:
        raise ValueError(
            "Must explicitly define atelast one Endpoint when " "two or more components are included in a composition."
        )
    return components, endpoints


class Composition(ServerMixin):
    """Create a composition which define computations / endpoints to create & run.

    Any number of components are accepted, which may have aribtrary connections
    between them. The final path through the component/connection DAG is determined
    by the root/terminal node position as specified by endpoint input/outputs keys.

    If only ONE component is provided, there is no need to create an Endpoint object.
    The library will generate a fully connected input/ouput endpoint for the one
    component with the `route` name set by the name of the method the `@expose`
    decorator is applied to.

    Parameters
    ----------
    kwargs
        Assignment of human readable names to ``ModelComponent`` and ``Endpoint``
        instances. If more than one ``ModelComponent`` is passed, an ``Endpoint``
        is needed as well.

    Warnings
    --------
    - This is a Work In Progress interface!

    Todo
    ----
    *  Move to connection components together at the composition level
    *  We plan to add some user-facing API to the ``Composition`` object
       which provides introspection of components, endpoints, etc.
    *  We plan to add some user-facing API to the ``Composition`` object
       which allows for modification of the composition.
    """

    _uid_comps: Dict[str, ModelComponent]
    _uid_names_map: Dict[str, str]
    _name_endpoints: Dict[str, Endpoint]
    _connections: List[Connection]
    _name_ep_protos: Dict[str, EndpointProtocol]
    DEBUG: bool
    TESTING: bool

    def __init__(
        self,
        *,
        DEBUG: bool = False,
        TESTING: bool = False,
        **kwargs: Union[ModelComponent, Endpoint],
    ):
        self.DEBUG = DEBUG
        self.TESTING = TESTING

        kwarg_comps, kwarg_endpoints = _parse_composition_kwargs(**kwargs)
        self._name_endpoints = kwarg_endpoints
        self._uid_comps = {v.uid: v for v in kwarg_comps.values()}
        self._uid_names_map = {v.uid: k for k, v in kwarg_comps.items()}

        self._connections = list(concat([c._flashserve_meta_.connections for c in kwarg_comps.values()]))

        if len(self._name_endpoints) == 0:
            comp = first(self.components.values())  # one element iterable
            ep_route = f"/{comp._flashserve_meta_.exposed.__name__}"
            ep_inputs = {k: f"{comp.uid}.inputs.{k}" for k in asdict(comp.inputs).keys()}
            ep_outputs = {k: f"{comp.uid}.outputs.{k}" for k in asdict(comp.outputs).keys()}
            ep = Endpoint(route=ep_route, inputs=ep_inputs, outputs=ep_outputs)
            self._name_endpoints[f"{comp._flashserve_meta_.exposed.__name__}_ENDPOINT"] = ep

        self._name_ep_protos = {}
        for ep_key, ep in self._name_endpoints.items():
            for ep_comp in itertools.chain(ep.inputs.values(), ep.outputs.values()):
                uid, argtype, name = ep_comp.split(".")
                if uid not in self.components:
                    raise AttributeError(f"{uid} not found. Expected one of {self.components.keys()}")
                try:
                    _ = getattr(getattr(self.components[uid], f"{argtype}"), name)
                except AttributeError:
                    raise AttributeError(f"uid={uid}, argtype={argtype}, name={name}")

            self._name_ep_protos[ep_key] = EndpointProtocol(name=ep_key, endpoint=ep, components=self.components)

    @property
    def endpoints(self) -> Dict[str, Endpoint]:
        return self._name_endpoints

    @property
    def endpoint_protocols(self) -> Dict[str, EndpointProtocol]:
        return self._name_ep_protos

    @property
    def connections(self) -> List[Connection]:
        return self._connections

    @property
    def components(self) -> Dict[str, ModelComponent]:
        return self._uid_comps

    @property
    def component_uid_names(self) -> Dict[str, str]:
        return self._uid_names_map
