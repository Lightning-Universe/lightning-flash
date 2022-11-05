from dataclasses import dataclass, field, fields
from functools import partial, wraps
from keyword import iskeyword
from types import FunctionType, MethodType
from typing import Dict, List, Sequence, Tuple, Union
from uuid import uuid4

from flash.core.serve.core import Connection, make_param_dict, make_parameter_container, ParameterContainer, Servable
from flash.core.serve.types.base import BaseType
from flash.core.serve.utils import fn_outputs_to_keyed_map
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _SERVE_TESTING

# Skip doctests if requirements aren't available
if not _SERVE_TESTING:
    __doctest_skip__ = ["*"]

if _CYTOOLZ_AVAILABLE:
    from cytoolz import compose
    from cytoolz import get as cytoolz_get
else:
    compose, cytoolz_get = None, None


@dataclass(unsafe_hash=True)
class UnboundMeta:
    __slots__ = ("exposed", "inputs", "outputs")

    exposed: Union[FunctionType, MethodType]
    inputs: Dict[str, BaseType]
    outputs: Dict[str, BaseType]


@dataclass(unsafe_hash=True)
class BoundMeta(UnboundMeta):

    models: Union[List["Servable"], Tuple["Servable", ...], Dict[str, "Servable"]]
    uid: str = field(default_factory=lambda: uuid4().hex, init=False)
    out_attr_dict: ParameterContainer = field(default=None, init=False)
    inp_attr_dict: ParameterContainer = field(default=None, init=False)
    dsk: Dict[str, tuple] = field(default_factory=dict, init=False)

    def __post_init__(self):
        i_pdict, o_pdict = make_param_dict(self.inputs, self.outputs, self.uid)
        self.inp_attr_dict = make_parameter_container(i_pdict)
        self.out_attr_dict = make_parameter_container(o_pdict)

        _dsk_func_inputs = []
        for k, datatype in self.inputs.items():
            _dsk_func_inputs.append(f"{self.uid}.inputs.{k}")
            self.dsk[f"{self.uid}.inputs.{k}"] = (
                datatype.packed_deserialize,
                f"{self.uid}.inputs.{k}.serial",
            )

        self.dsk[f"{self.uid}.funcout"] = (
            # inline _exposed_fn run with 'outputs_to_keymap_fn' since
            # it is a cheap transformation we need to do every time.
            compose(partial(fn_outputs_to_keyed_map, self.outputs.keys()), self.exposed),
            *_dsk_func_inputs,
        )

        for k, datatype in self.outputs.items():
            self.dsk[f"{self.uid}.outputs.{k}"] = (
                partial(cytoolz_get, k),
                f"{self.uid}.funcout",
            )
            self.dsk[f"{self.uid}.outputs.{k}.serial"] = (
                datatype.serialize,
                f"{self.uid}.outputs.{k}",
            )

    @property
    def connections(self) -> Sequence["Connection"]:
        connections = []
        for fld in fields(self.inp_attr_dict):
            connections.extend(getattr(self.inp_attr_dict, fld.name).connections)
        for fld in fields(self.out_attr_dict):
            connections.extend(getattr(self.out_attr_dict, fld.name).connections)
        return connections


def _validate_expose_inputs_outputs_args(kwargs: Dict[str, BaseType]):
    """Checks format & type of arguments passed to `@expose` inputs/outputs parameters.

    Parameters
    ----------
    kwargs
        dict of inputs to check.

    Raises
    ------
    SyntaxError
        If the inputs / outputs exposed dict are invalid:
        *  Keys must be str type
    TypeError
        If the inputs / outputs exposed dict are invalid:
        *  values must be instance of `BaseType`.
    ValueError
        If the inputs / output dicts are not of length >= 1
    RuntimeError:
        If input keys passed to `@expose` do not match the corresponding
        (decorated) method parameter names. (TODO!!)

    Examples
    --------
    >>> from flash.core.serve.types import Number
    >>> inp = {'hello': Number()}
    >>> out = {'out': Number()}
    >>> _validate_expose_inputs_outputs_args(inp)
    >>> _validate_expose_inputs_outputs_args(out)
    """
    if not isinstance(kwargs, dict):
        raise TypeError(f"`expose` values must be {dict}. recieved {kwargs}")

    if len(kwargs) < 1:
        raise ValueError(f"cannot set dict of length < 1 for field=`{field}`")

    for k, v in kwargs.items():
        if not k.isidentifier() or iskeyword(k):
            raise SyntaxError(f"`expose key={k} must be valid python attribute")
        if not isinstance(v, BaseType):
            raise TypeError(f"expose key {k}, v={v} must be subclass of {BaseType}")


def expose(inputs: Dict[str, BaseType], outputs: Dict[str, BaseType]):
    """Expose a function/method via a web API for serving model inference.

    The ``@expose`` decorator has two arguments, inputs and outputs, which
    describe how the inputs to predict are decoded from the request and how
    the outputs of predict are encoded to a response.

    Must decorate one (and only one) method when used within a subclass
    of ``ModelComponent``.

    Parameters
    ----------
    inputs
        accepts a dictionary mapping keys to decorated method parameter
        names (must be one to one mapping) with values corresponding to
        an instantiated specification of a Flash Serve Data Type (ie.
        ``Number()``, ``Image()``, ``Text()``, etc...)
    outputs
        accepts a dictionary mapping outputs of the decorated method to
        keys and data type (similar to inputs). However, unlike ``inputs``
        the output keys are less strict in their names. IF the method
        returns a dictionary, the keys must match one-to-one. However, if
        the method returns a sorted sequence (list / tuple) the keys can be
        arbitrary, so long as no reserved names are used (primarily python
        keywords). For result sequences, the order in which keys are defined
        maps to the appropriate element index in the result (ie.
        ``key 0 -> sequence[0]``, ``key 1 -> sequence[1]``, etc.)

    TODO
    ----
    *  Examples in the docstring.
    """
    _validate_expose_inputs_outputs_args(inputs)
    _validate_expose_inputs_outputs_args(outputs)

    def wrapper(fn):
        @wraps(fn)
        def wrapped(func):
            func.flashserve_meta = UnboundMeta(exposed=func, inputs=inputs, outputs=outputs)
            return func

        return wrapped(fn)

    return wrapper
