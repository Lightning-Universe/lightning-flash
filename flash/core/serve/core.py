import dataclasses
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import pytorch_lightning as pl
import torch

from flash.core.serve.types.base import BaseType
from flash.core.serve.utils import download_file
from flash.core.utilities.imports import _PYDANTIC_AVAILABLE, requires

if _PYDANTIC_AVAILABLE:
    from pydantic import FilePath, HttpUrl, parse_obj_as, ValidationError
else:
    FilePath, HttpUrl, parse_obj_as, ValidationError = None, None, None, None

# -------------------------------- Endpoint -----------------------------------


@dataclass
class Endpoint:
    """An endpoint maps a route and request/response payload to components.

    Parameters
    ----------
    route
        The API route name to construct as the servicing POST endpoint.
    inputs
        The full name of a component input. Typically, specified by just passing
        in the component parameter attribute (i.e.``component.inputs.foo``).
    outputs
        The full name of a component output. Typically, specified by just passing
        in the component parameter attribute (i.e.``component.outputs.bar``).
    """

    route: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]

    def __post_init__(self):
        if not isinstance(self.route, str):
            raise TypeError(
                f"route parameter must be type={str}, recieved " f"route={self.route} of type={type(self.route)}"
            )
        if not self.route.startswith("/"):
            raise ValueError("route must begin with a `slash` character (ie `/`).")

        for k in tuple(self.inputs.keys()):
            v = self.inputs[k]
            if not isinstance(v, (Parameter, str)):
                raise TypeError(f"inputs k={k}, v={v}, is not {Parameter} or {str}. type(v)={type(v)}")
            self.inputs[k] = str(v)

        for k in tuple(self.outputs.keys()):
            v = self.outputs[k]
            if not isinstance(v, (Parameter, str)):
                raise TypeError(f"k={k}, v={v}, type(v)={type(v)}")
            self.outputs[k] = str(v)


# -------------------------------- Servable ---------------------------------


class FlashServeScriptLoader:

    __slots__ = ("location", "instance")

    def __init__(self, location: FilePath):
        self.location = location
        self.instance = torch.jit.load(location)

    def __call__(self, *args, **kwargs):
        print(self.instance, args, kwargs)
        return self.instance(*args, **kwargs)


ServableValidArgs_T = Union[
    Tuple[Type[pl.LightningModule], Union[HttpUrl, FilePath]],
    Tuple[HttpUrl],
    Tuple[FilePath],
]


class Servable:
    """ModuleWrapperBase around a model object to enable serving at scale.

    Create a ``Servable`` from either (LM, LOCATION) or (LOCATION,)

    Parameters
    ----------
    *args
        A model class and path to the asset file (url or local file path) OR
        a singular path to a torchscript asset which can be loaded without the
        model class definition.
    download_path
        Optional url to download a model from.

    TODO
    ----
    *  How to handle ``__init__`` args for ``torch.nn.Module``
    *  How to handle ``__init__`` args not recorded in hparams of ``pl.LightningModule``
    """

    @requires("serve")
    def __init__(
        self,
        *args: ServableValidArgs_T,
        download_path: Optional[Path] = None,
        script_loader_cls: Type[FlashServeScriptLoader] = FlashServeScriptLoader,
    ):
        try:
            loc = args[-1]  # last element in args is always loc
            parsed = parse_obj_as(ServableValidArgs_T, tuple(args))
        except ValidationError:
            if args[0].__qualname__ != script_loader_cls.__qualname__:
                raise
            parsed = [script_loader_cls, parse_obj_as(Union[HttpUrl, FilePath], loc)]

        if isinstance(parsed[-1], Path):
            f_path = loc
        else:
            f_path = download_file(loc, download_path=download_path)

        if len(args) == 2 and args[0].__qualname__ != script_loader_cls.__qualname__:
            # if this is a class and path/url...
            klass = args[0]
            instance = klass.load_from_checkpoint(f_path)
        else:
            # if this is just a path/url
            klass = script_loader_cls
            instance = klass(f_path)

        self.instance = instance

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    def __repr__(self):
        return repr(self.instance)


# ------------------ Connections & Parameters (internal) ----------------------


class Connection(NamedTuple):
    """A connection maps one output to one input.

    This is a self-contained data structure, which when given in the context of
    the other components in a composition, will map input/output keys/indices
    between components.

    Warnings
    --------
    * This data structure should not be instantiated directly! The
      class_methods attached to the class are the indended mechanisms to create
      a new instance.
    """

    source_component: str
    target_component: str
    source_key: str
    target_key: str

    def __repr__(self):  # pragma: no cover
        return f"Connection({str(self)})"

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        if cycle:
            return
        res = (
            f"Connection("
            f"{self.source_component}.outputs.{self.source_key} >> "
            f"{self.target_component}.inputs.{self.target_key})"
        )
        p.text(res)

    def __str__(self):
        return (
            f"{self.source_component}.outputs.{self.source_key} >> " f"{self.target_component}.inputs.{self.target_key}"
        )


@dataclass
class Parameter:
    """Holder class for each grid type of component and connections from those to the types of other components.

    Parameters
    ----------
    name
        Name of the parameter. It's same as the dictionary key from `expose`
    datatype
        Grid type object
    component_uid
        Which component this type is associated with
    position
        Position in the while exposing it i.e `inputs` or `outputs`
    """

    name: str
    datatype: BaseType
    component_uid: str
    position: str
    connections: List["Connection"] = field(default_factory=list, init=False, repr=False)

    def __str__(self):
        return f"{self.component_uid}.{self.position}.{self.name}"

    def __terminate_invalid_connection_request(self, other: "Parameter", dunder_meth_called: str) -> None:
        """verify that components can be composed.

        Parameters
        ----------
        other
            object passed into the bitshift operator. We verify if is a
            ``Parameter`` class and that is not the type of the same component
        dunder_meth_called: str
            one of ['__lshift__', '__rshift__']. we need to know the
            directionality of the bitshift method called when we verify
            that the directionality of the dag is always outputs -> inputs.

        Raises
        ------
        TypeError, RuntimeError
            if the verification fails, we throw an exception to stop the
            connection from being created.
        """
        # assert this is actually a class object we can compare against.
        if not isinstance(other, self.__class__) or (other.__class__ != self.__class__):
            raise TypeError(f"Can only Compose another `Parameter` class, not {type(other)}")

        # assert not same instance
        if id(other) == id(self):
            raise RuntimeError("Cannot compose a parameters of same components")

        # assert bitshift directionality is acceptable for source/target map
        source = other if dunder_meth_called == "__lshift__" else self
        target = self if dunder_meth_called == "__lshift__" else other
        if source.position != "outputs":
            raise TypeError(
                f"A data source component can only provide a target with data listed "
                f"as ``output``. source component: `{source.component_uid}` "
                f"key: `{source.name}`"
            )
        if target.position != "inputs":
            raise TypeError(
                f"A data target component can only accept data into keys listed as "
                f"`inputs`. components: source=`{str(source)}` target={str(target)}"
            )
        if source.component_uid == target.component_uid:
            raise RuntimeError(
                f"Cannot create cycle by creating connection between outputs and "
                f"inputs of a single component. source component: `{source.component_uid}`"
            )

    def __lshift__(self, other: "Parameter"):
        """Implements composition connecting Parameter << Parameter."""
        self.__terminate_invalid_connection_request(other, "__lshift__")
        con = Connection(
            source_component=other.component_uid,
            target_component=self.component_uid,
            source_key=other.name,
            target_key=self.name,
        )
        self.connections.append(con)

    def __rshift__(self, other: "Parameter"):
        """Implements composition connecting Parameter >> Parameter."""
        self.__terminate_invalid_connection_request(other, "__rshift__")
        con = Connection(
            source_component=self.component_uid,
            target_component=other.component_uid,
            source_key=self.name,
            target_key=other.name,
        )
        self.connections.append(con)


class DictAttrAccessBase:
    def __grid_fields__(self) -> Iterator[str]:
        for field in dataclasses.fields(self):  # noqa F402
            yield field.name

    def __getitem__(self, item) -> Parameter:
        return getattr(self, item)

    def __contains__(self, item):
        return bool(getattr(self, item, False))

    def __len__(self):
        return len(tuple(self.__grid_fields__()))

    def __iter__(self):
        yield from self.__grid_fields__()


ParameterContainer = TypeVar("ParameterContainer", bound=DictAttrAccessBase)


# skipcq: PYL-W1401, PYL-W0621
def make_parameter_container(data: Dict[str, Parameter]) -> ParameterContainer:
    """Create dotted dict lookup class from parameter map.

    Parameters
    ----------
    data
        mapping for ``parameter_name -> 'Parameter' instance``

    Returns
    -------
    ParameterContainer
        A representation of the parameter data dict with keys accessible via
        ``dotted`` attribute lookup.

    Notes
    -----
    *  parameter name must be valid python attribute (identifier) and
       cannot be a builtin keyword. input names should have been validated
       by this point.
    """
    dataclass_fields = [(param_name, type(param)) for param_name, param in data.items()]
    ParameterContainer = make_dataclass(
        "ParameterContainer",
        dataclass_fields,
        bases=(DictAttrAccessBase,),
        frozen=True,
        unsafe_hash=True,
    )
    return ParameterContainer(**data)


def make_param_dict(
    inputs: Dict[str, BaseType], outputs: Dict[str, BaseType], component_uid: str
) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
    """Convert exposed input/outputs parameters / dtypes to parameter objects.

    Returns
    -------
    Tuple[Dict[str, Parameter], Dict[str, Parameter]]
        Element[0] == Input parameter dict
        Element[1] == Output parameter dict.
    """
    flashserve_inp_params, flashserve_out_params = {}, {}
    for inp_key, inp_dtype in inputs.items():
        flashserve_inp_params[inp_key] = Parameter(
            name=inp_key, datatype=inp_dtype, component_uid=component_uid, position="inputs"
        )

    for out_key, out_dtype in outputs.items():
        flashserve_out_params[out_key] = Parameter(
            name=out_key, datatype=out_dtype, component_uid=component_uid, position="outputs"
        )
    return flashserve_inp_params, flashserve_out_params
