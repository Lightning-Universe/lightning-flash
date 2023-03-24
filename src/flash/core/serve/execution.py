from collections import defaultdict
from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

from flash.core.serve.dag.optimization import cull, functions_of, inline_functions
from flash.core.serve.dag.rewrite import RewriteRule, RuleSet
from flash.core.serve.dag.task import flatten, get_deps, getcycle, isdag, toposort
from flash.core.serve.dag.utils import funcname
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _PYDANTIC_AVAILABLE

if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel
else:
    BaseModel = object

if _CYTOOLZ_AVAILABLE:
    from cytoolz import identity, merge, valmap
else:
    identity, merge, valmap = None, None, None

if TYPE_CHECKING:  # pragma: no cover
    from flash.core.serve.component import ModelComponent
    from flash.core.serve.composition import EndpointProtocol
    from flash.core.serve.core import Connection


class EndpointProtoJSON(BaseModel):
    name: str
    route: str
    payload_key_dsk_task: Dict[str, str]
    result_key_dsk_task: Dict[str, str]


class ComponentJSON(BaseModel):
    component_dependencies: Dict[str, Dict[str, Set[str]]]
    component_dependents: Dict[str, Dict[str, Set[str]]]
    component_funcnames: Dict[str, Dict[str, Tuple[str, ...]]]
    connections: List[Dict[str, str]]


class MergedJSON(BaseModel):
    dependencies: Dict[str, Set[str]]
    dependents: Dict[str, Set[str]]
    funcnames: Dict[str, Tuple[str, ...]]
    connections: List[Dict[str, str]]
    endpoint: EndpointProtoJSON


@dataclass
class TaskComposition:
    """Contains info which can be used to setup / run a computation.

    Attributes
    ----------
    dsk
        The computation graph. Contains mapping of task key names ->
        callable & dependency tuples
    sortkeys
        Topologically sorted ordering of DAG execution path
    get_keys
        The keys which are results of the DAG for this endpoint
    ep_dsk_input_keys
        map of endpoint input payload key to input dsk key
    ep_dsk_output_keys
         map of endpoint ouput (results) key to output task key
    pre_optimization_dsk
        Merged component `_dsk` subgraphs (without payload / result
        mapping or connections applied.)
    """

    __slots__ = (
        "dsk",
        "sortkeys",
        "get_keys",
        "ep_dsk_input_keys",
        "ep_dsk_output_keys",
        "pre_optimization_dsk",
    )

    dsk: Dict[str, tuple]
    sortkeys: List[str]
    get_keys: List[str]
    ep_dsk_input_keys: Dict[str, str]
    ep_dsk_output_keys: Dict[str, str]
    pre_optimization_dsk: Dict[str, tuple]


@dataclass
class UnprocessedTaskDask:
    """Unconnected extraction of task dsk and payload / results key info.

    By "unconnected" we mean, the connections between components and
    inputs / outputs of endpoints has not been applied to the DAG
    representation.

    Attributes
    ----------
    component_dsk
        component `_dsk` subgraphs (without payload / result mapping
        or connections applied) with a top level "component" name key.
    merged_dsk
        Merged component `_dsk` subgraphs (without payload / result
        mapping or connections applied.)
    payload_tasks_dsk
        dsk of input payload key to input task
    payload_dsk_map
        map of input payload key to input dsk key
    result_tasks_dsk
        dsk of ouput (results) key to output task
    res_dsk_map
        map of ouput (results) key to output task key
    output_keys
        keys to get as results
    """

    __slots__ = (
        "component_dsk",
        "merged_dsk",
        "payload_tasks_dsk",
        "payload_dsk_map",
        "result_tasks_dsk",
        "result_dsk_map",
        "output_keys",
    )

    component_dsk: Dict[str, Dict[str, tuple]]
    merged_dsk: Dict[str, tuple]
    payload_tasks_dsk: Dict[str, tuple]
    payload_dsk_map: Dict[str, str]
    result_tasks_dsk: Dict[str, tuple]
    result_dsk_map: Dict[str, str]
    output_keys: List[str]


def _process_initial(
    endpoint_protocol: "EndpointProtocol", components: Dict[str, "ModelComponent"]
) -> UnprocessedTaskDask:
    """Extract task dsk and payload / results keys and return computable form.

    Parameters
    ----------
    endpoint_protocol
        endpoint protocol definition for the variation of the DAG which
        is currently being evaluated.
    components
        Mapping of component name -> component class definitions which
        contain independent subgraph task dsks'.

    Returns
    -------
    UnprocessedTaskDask
    """

    # mapping payload input keys -> serialized keys / tasks
    payload_dsk_key_map = {
        payload_key: f"{input_key}.serial" for payload_key, input_key in endpoint_protocol.dsk_input_key_map.items()
    }
    payload_input_tasks_dsk = {
        input_dsk_key: (identity, payload_key) for payload_key, input_dsk_key in payload_dsk_key_map.items()
    }

    # mapping result keys -> serialize keys / tasks
    res_dsk_key_map = {
        result_key: f"{output_key}.serial" for result_key, output_key in endpoint_protocol.dsk_output_key_map.items()
    }
    result_output_tasks_dsk = {
        result_key: (identity, output_dsk_key) for result_key, output_dsk_key in res_dsk_key_map.items()
    }
    output_keys = list(res_dsk_key_map.keys())

    # need check to prevent cycle error
    _payload_keys = set(payload_dsk_key_map.keys())
    _result_keys = set(res_dsk_key_map.keys())
    if not _payload_keys.isdisjoint(_result_keys):
        raise KeyError(
            f"Request payload keys `{_payload_keys}` and response keys `{_result_keys}` "
            f"names cannot intersectt. keys: `{_payload_keys.intersection(_result_keys)}` "
            f"must be renamed in either `inputs` or `outputs`. "
        )

    component_dsk = merge(valmap(attrgetter("_flashserve_meta_.dsk"), components))
    merged_dsk = merge(*(dsk for dsk in component_dsk.values()))

    return UnprocessedTaskDask(
        component_dsk=component_dsk,
        merged_dsk=merged_dsk,
        payload_tasks_dsk=payload_input_tasks_dsk,
        payload_dsk_map=payload_dsk_key_map,
        result_tasks_dsk=result_output_tasks_dsk,
        result_dsk_map=res_dsk_key_map,
        output_keys=output_keys,
    )


def build_composition(
    endpoint_protocol: "EndpointProtocol",
    components: Dict[str, "ModelComponent"],
    connections: List["Connection"],
) -> "TaskComposition":
    r"""Build a composed graph.

    Notes on easy sources to introduce bugs.

    ::

            Input Data
        --------------------
            a  b  c   d
            |  |  |   | \\
             \ | / \  |  ||
              C_2   C_1  ||
            /  |     | \ //
           /   |    /   *
        RES_2  |   |   // \
               |   |  //   RES_1
                \  | //
                C_2_1
                  |
                RES_3
        ---------------------
              Output Data

    Because there are connections between ``C_1 -> C_2_1`` and
    ``C_2 -> C_2_1`` we can eliminate the ``serialize <-> deserialize``
    tasks for the data transfered between these components. We need to be
    careful to not eliminate the ``serialize`` or ``deserialize`` tasks
    entirely though. In the case shown above, it is apparent ``RES_1`` &
    ``RES_2``. still need the ``serialize`` function, but the same also applies
    for ``deserialize``. Consider the example below with the same composition &
    connections as above:

    ::
            Input Data
        --------------------
            a  b  c   d
            |  |  |   | \\
             \ | /| \ |  \\
              C_2 |  C_1  ||
            /  |  |   @\  ||
           /   |  |   @ \ //
        RES_2  |  |  @   *
               |  | @  // \
                \ | @ //   RES_1
                 C_2_1
                  |
                RES_3
        ---------------------
              Output Data

    Though we are using the same composition, the endpoints have been changed so
    that the previous result of ``C_1``-> ``C_2_1`` is now being provided by
    input ``c``. However, there is still a connection between ``C_1`` and
    ``C_2_1`` which is denoted by the ``@`` symbols... Though the first
    example (shown at the top of this docstring) would be able to eliminate
    ``C_2_1 deserailize``from ``C_2`` / ``C_1``, we see here that since
    endpoints define the path through the DAG, we cannot eliminate them
    entirely either.
    """
    initial_task_dsk = _process_initial(endpoint_protocol, components)

    dsk_tgt_src_connections = {}
    for connection in connections:
        source_dsk = f"{connection.source_component}.outputs.{connection.source_key}"
        target_dsk = f"{connection.target_component}.inputs.{connection.target_key}"
        # value of target key is mapped one-to-one from value of source
        dsk_tgt_src_connections[target_dsk] = (identity, source_dsk)

    rewrite_ruleset = RuleSet()
    for dsk_payload_target_serial in initial_task_dsk.payload_tasks_dsk.keys():
        dsk_payload_target, _serial_ident = dsk_payload_target_serial.rsplit(".", maxsplit=1)
        if _serial_ident != "serial":
            raise RuntimeError(
                f"dsk_payload_target_serial={dsk_payload_target_serial}, "
                f"dsk_payload_target={dsk_payload_target}, _serial_ident={_serial_ident}"
            )
        if dsk_payload_target in dsk_tgt_src_connections:
            # This rewrite rule ensures that exposed inputs are able to replace inputs
            # coming from connected components. If the payload keys are mapped in a
            # connection, replace the connection with the payload deserialize function.
            lhs = dsk_tgt_src_connections[dsk_payload_target]
            rhs = initial_task_dsk.merged_dsk[dsk_payload_target]
            rule = RewriteRule(lhs, rhs, vars=())
            rewrite_ruleset.add(rule)

    io_subgraphs_merged = merge(
        initial_task_dsk.merged_dsk,
        dsk_tgt_src_connections,
        initial_task_dsk.result_tasks_dsk,
        initial_task_dsk.payload_tasks_dsk,
    )

    # apply rewrite rules
    rewritten_dsk = valmap(rewrite_ruleset.rewrite, io_subgraphs_merged)

    # We perform a significant optimization here by culling any tasks which
    # have been made redundant by the rewrite rules, or which don't exist
    # on a path which is required for computation of the endpoint outputs
    culled_dsk, culled_deps = cull(rewritten_dsk, initial_task_dsk.output_keys)
    _verify_no_cycles(culled_dsk, initial_task_dsk.output_keys, endpoint_protocol.name)

    # as an optimization, we inline the `one_to_one` functions, into the
    # execution of their dependency. Since they are so cheap, there's no
    # need to spend time sending off a task to perform them.
    inlined = inline_functions(
        culled_dsk,
        initial_task_dsk.output_keys,
        fast_functions=[identity],
        inline_constants=True,
        dependencies=culled_deps,
    )
    inlined_culled_dsk, inlined_culled_deps = cull(inlined, initial_task_dsk.output_keys)
    _verify_no_cycles(inlined_culled_dsk, initial_task_dsk.output_keys, endpoint_protocol.name)

    # pe-run topological sort of tasks so it doesn't have to be
    # recomputed upon every request.
    toposort_keys = toposort(inlined_culled_dsk)

    # construct results
    res = TaskComposition(
        dsk=inlined_culled_dsk,
        sortkeys=toposort_keys,
        get_keys=initial_task_dsk.output_keys,
        ep_dsk_input_keys=initial_task_dsk.payload_dsk_map,
        ep_dsk_output_keys=initial_task_dsk.result_dsk_map,
        pre_optimization_dsk=initial_task_dsk.merged_dsk,
    )
    return res


def _verify_no_cycles(dsk: Dict[str, tuple], out_keys: List[str], endpoint_name: str):
    if not isdag(dsk, keys=out_keys):
        cycle = getcycle(dsk, keys=out_keys)
        raise RuntimeError(
            f"Cycle detected when attepting to build DAG for endpoint: "
            f"`{endpoint_name}`. This cycle is formed by connections between "
            f"the following nodes: {cycle}"
        )


def connections_from_components_map(components: Dict[str, "ModelComponent"]) -> List[Dict[str, str]]:
    dsk_connections = []
    for con in flatten([comp._flashserve_meta_.connections for comp in components.values()]):
        # value of target key is mapped one-to-one from value of source
        dsk_connections.append(con._asdict())
    return dsk_connections


def endpoint_protocol_content(ep_proto: "EndpointProtocol") -> "EndpointProtoJSON":
    ep_proto_payload_dsk_key_map = valmap(lambda x: f"{x}.serial", ep_proto.dsk_input_key_map)
    ep_proto_result_key_dsk_map = valmap(lambda x: f"{x}.serial", ep_proto.dsk_output_key_map)

    return EndpointProtoJSON(
        name=ep_proto.name,
        route=ep_proto.route,
        payload_key_dsk_task=ep_proto_payload_dsk_key_map,
        result_key_dsk_task=ep_proto_result_key_dsk_map,
    )


def merged_dag_content(ep_proto: "EndpointProtocol", components: Dict[str, "ModelComponent"]) -> "MergedJSON":
    init = _process_initial(ep_proto, components)
    dsk_connections = connections_from_components_map(components)
    epjson = endpoint_protocol_content(ep_proto)

    merged = {**init.merged_dsk, **init.payload_tasks_dsk}
    dependencies, _ = get_deps(merged)
    merged_proto = defaultdict(list)
    for task_name, task in merged.items():
        for parent in dependencies[task_name]:
            merged_proto[task_name].append(parent)

    for request_name, task_key in init.payload_dsk_map.items():
        cluster, *_ = task_key.split(".")
        merged_proto[task_key[: -len(".serial")]].append(task_key)
        merged_proto[task_key].append(request_name)
    merged_proto = dict(merged_proto)

    dependencies, dependents = get_deps(merged_proto)
    dependents = dict(dependents)
    functions_merged = valmap(functions_of, merged)
    function_names_merged = {k: tuple(map(funcname, v)) for k, v in functions_merged.items()}

    return MergedJSON(
        dependencies=dependencies,
        dependents=dependents,
        funcnames=function_names_merged,
        connections=dsk_connections,
        endpoint=epjson,
    )


def component_dag_content(components: Dict[str, "ModelComponent"]) -> "ComponentJSON":
    dsk_connections = connections_from_components_map(components)
    comp_dependencies, comp_dependents, comp_funcnames = {}, {}, {}

    for comp_name, comp in components.items():
        functions_comp = valmap(functions_of, comp._flashserve_meta_.dsk)
        function_names_comp = {k: sorted(set(map(funcname, v))) for k, v in functions_comp.items()}
        comp_funcnames[comp_name] = function_names_comp
        _dependencies, _dependents = get_deps(comp._flashserve_meta_.dsk)
        _dependents = dict(_dependents)
        comp_dependencies[comp_name] = _dependencies
        comp_dependents[comp_name] = _dependents

    return ComponentJSON(
        component_dependencies=comp_dependencies,
        component_dependents=comp_dependents,
        component_funcnames=comp_funcnames,
        connections=dsk_connections,
    )
