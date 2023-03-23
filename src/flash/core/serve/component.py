import inspect
from dataclasses import replace
from functools import wraps
from typing import Dict, List, Optional, Tuple, Type, Union

from torch import nn

from flash.core.serve.core import ParameterContainer, Servable
from flash.core.serve.decorators import BoundMeta, UnboundMeta
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _SERVE_AVAILABLE, requires

if _CYTOOLZ_AVAILABLE:
    from cytoolz import first, isiterable, valfilter
else:
    first, isiterable, valfilter = None, None, None

# ------------------- Validation Funcs and Pydantic Models --------------------

_FLASH_SERVE_RESERVED_NAMES = ("inputs", "outputs", "uid")


def _validate_exposed_input_parameters_valid(instance):
    """Raises RuntimeError if exposed parameters != method argument names."""
    spec = inspect.getfullargspec(instance._flashserve_meta_.exposed)

    exposed_args = spec.args[1:]  # do not include `self` arg
    if spec.varargs:
        exposed_args.extend(spec.varargs)
    if spec.varkw:
        exposed_args.extend(spec.varkw)
    if spec.kwonlyargs:
        exposed_args.extend(spec.kwonlyargs)

    diff = set(exposed_args).symmetric_difference(instance._flashserve_meta_.inputs.keys())
    if len(diff) > 0:
        raise RuntimeError(
            f"Methods decorated by `@expose` must list all method arguments in `inputs` "
            f"parameter passed to `expose`. Expected: exposed method args = `{exposed_args}` "
            f"recieved input keys passed to `expose` = `{instance._flashserve_meta_.inputs.keys()}`. "
            f"Difference = `{diff}`."
        )


def _validate_subclass_init_signature(cls: Type["ModelComponent"]):
    """Raises SyntaxError if the __init__ method is not formatted correctly.

    Expects arguments: ['self', 'models', Optional['config']]

    Parameters
    ----------
    cls
        class to perform the analysis on

    Raises
    ------
    SyntaxError
        If parameters are not specified correctly.
    """
    params = inspect.signature(cls.__init__).parameters
    if len(params) > 3:
        raise SyntaxError(
            "__init__ can only have 1 or 2 parameters. Must conform to "
            "specification: (`'self', 'model', Optional['config']`)"
        )
    for idx, param in enumerate(params.keys()):
        if (idx == 1) and (param != "model"):
            raise SyntaxError(f"__init__ must set 'model' as first param, not `{param}`")
        if (idx == 2) and (param != "config"):
            raise SyntaxError(f"__init__ can only set 'config' as second param, not `{param}`")


_ServableType = Union[Servable, nn.Module]
_Servable_t = (Servable, nn.Module)


def _validate_model_args(
    args: Union[_ServableType, List[_ServableType], Tuple[_ServableType, ...], Dict[str, _ServableType]]
) -> None:
    """Validator for machine learning models.

    Parameters
    ----------
    args
        model args passed into ``__init__`` of ``ModelComponent``

    Raises
    ------
    ValueError
        If an empty iterable is passed as the model argument
    TypeError
        If the args do not contain properly formatted model refences
    """
    if isiterable(args) and len(args) == 0:
        raise ValueError(f"Iterable args={args} must have length >= 1")

    if isinstance(args, (list, tuple)):
        if not all(isinstance(x, _Servable_t) for x in args):
            raise TypeError(f"One of arg in args={args} is not type {_Servable_t}")
    elif isinstance(args, dict):
        if not all(isinstance(x, str) for x in args.keys()):
            raise TypeError(f"One of keys in args={args.keys()} is not type {str}")
        if not all(isinstance(x, _Servable_t) for x in args.values()):
            raise TypeError(f"One of values in args={args} is not type {_Servable_t}")
    elif not isinstance(args, _Servable_t):
        raise TypeError(f"Args must be instance, list/tuple, or mapping of {_Servable_t}")


def _validate_config_args(config: Optional[Dict[str, Union[str, int, float, bytes]]]) -> None:
    """Validator for the configuration.

    Parameters
    ----------
    config
        configuration arguments passed into ``__init__`` of
        ``ModelComponent``

    Raises
    ------
    TypeError
        If ``config`` is not a dict.
    TypeError
        If ``config`` is a dict with invalid key/values
    ValueError
        If ``config`` is a dict with 0 arguments
    """
    if config is None:
        return

    if not isinstance(config, dict):
        raise TypeError(f"Config must be {dict}. Recieved config={config}")

    if len(config) == 0:
        raise ValueError("cannot set dict of length < 1 for `config`")

    for k, v in config.items():
        if not isinstance(k, str):
            raise TypeError(f"config key={k} != {str} type")
        if not isinstance(v, (str, bytes, int, float)):
            raise TypeError(f"config val k={k}, v={v} != {(str, bytes, int, float)} type")


# ------------------- ModelComponent and Metaclass Validators------------------------


class FlashServeMeta(type):
    """We keep a mapping of externally used names to classes."""

    @requires("serve")
    def __new__(cls, name, bases, namespace):
        # create new instance of cls in order to apply any @expose class decorations.
        _tmp_cls = super().__new__(cls, name, bases, namespace)

        # determine which methods have been exposed.
        ex_meths = valfilter(lambda x: hasattr(x, "flashserve_meta"), _tmp_cls.__dict__)
        if _tmp_cls.__name__ != "ModelComponent":
            if len(ex_meths) != 1:
                raise SyntaxError(
                    f"`@expose` decorator must be applied to one (and only one) method in a "
                    f"class class=`{_tmp_cls.__name__}` detected n=`{len(ex_meths)}` "
                    f"decorations on method_names=`{list(ex_meths.keys())}`"
                )

            # alter namespace to insert flash serve info as bound components of class.
            exposed = first(ex_meths.values())
            namespace["_flashserve_meta_"] = exposed.flashserve_meta
            namespace["__call__"] = wraps(exposed)(
                exposed,
            )

        new_cls = super().__new__(cls, name, bases, namespace)
        if new_cls.__name__ != "ModelComponent":
            # If user defined class, validate.
            _validate_subclass_init_signature(new_cls)
            if set(_FLASH_SERVE_RESERVED_NAMES).intersection(namespace):
                raise TypeError(
                    f"Subclasses of {bases[-1]} are not allowed to define bound methods/"
                    f"attrs named: `{set(_FLASH_SERVE_RESERVED_NAMES).intersection(namespace)}`"
                )
        return new_cls

    def __call__(cls, *args, **kwargs):
        """Customize steps taken during class creation / initalization.

        super().__call__() within metaclass means: return instance created by calling metaclass __prepare__ -> __new__
        -> __init__
        """
        klass = super().__call__(*args, **kwargs)
        klass._flashserve_meta_ = replace(klass._flashserve_meta_)
        _validate_exposed_input_parameters_valid(klass)
        klass.__flashserve_init__(*args, **kwargs)
        return klass


if _SERVE_AVAILABLE:

    class ModelComponent(metaclass=FlashServeMeta):
        """Represents a computation which is decorated by `@expose`.

        A component is how we represent the main unit of work; it is a set of
        evaluations which involve some input being passed through some set of
        functions to generate some set of outputs.

        To specify a component, we record things like: its name, source file
        assets, configuration args, model source assets, etc. The
        specification must be YAML serializable and loadable to/from a fully
        initialized instance. It must contain the minimal set of information
        necessary to find and initialize its dependencies (assets) and itself.
        """

        _flashserve_meta_: Optional[Union[BoundMeta, UnboundMeta]] = None

        def __flashserve_init__(self, models, *, config=None):
            """Do a bunch of setup.

            instance's __flashserve_init__ calls subclass __init__ in turn.
            """
            _validate_model_args(models)
            _validate_config_args(config)

            try:
                self.__init__(models, config=config)
            except TypeError:
                self.__init__(models)

            bound_fn = getattr(self, self._flashserve_meta_.exposed.__name__)
            self.__call__ = bound_fn
            self._flashserve_meta_ = BoundMeta(
                exposed=bound_fn,
                inputs=self._flashserve_meta_.inputs,
                outputs=self._flashserve_meta_.outputs,
                models=models,
            )

            return self

        @property
        def inputs(self) -> ParameterContainer:
            return self._flashserve_meta_.inp_attr_dict

        @property
        def outputs(self) -> ParameterContainer:
            return self._flashserve_meta_.out_attr_dict

        @property
        def uid(self) -> str:
            return self._flashserve_meta_.uid

else:
    ModelComponent = object
