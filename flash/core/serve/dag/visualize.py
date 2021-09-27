from contextlib import suppress
from io import BytesIO

from flash.core.serve.dag.task import get_deps
from flash.core.serve.execution import TaskComposition

with suppress(ImportError):
    import graphviz


def _dag_to_graphviz(dag, dependencies, request_data, response_data, *, no_optimization=False):
    if not graphviz:  # pragma: no cover
        raise ImportError("Visualizing graphs requires graphviz")

    graph_attr = {"rankdir": "BT"}
    g = graphviz.Digraph(graph_attr=graph_attr)

    for task_name, task in dag.items():
        if task_name not in response_data:
            # not an endpoint result.
            cluster, *_ = task_name.split(".")
            with g.subgraph(name=f"cluster_{cluster}") as c:
                c.node(task_name, task_name, shape="rectangle")
                c.attr(label=f"Component: {cluster}", color="blue")
        else:
            # an endpoint result
            g.node(task_name, task_name, shape="rectangle")

        for parent in dependencies[task_name]:
            g.edge(parent, task_name)

    if no_optimization:
        return g

    for request_name, task_key in request_data.items():
        cluster, *_ = task_key.split(".")
        g.node(request_name, request_name, shape="oval")
        with g.subgraph(name=f"cluster_{cluster}") as c:
            c.node(task_key, task_key, shape="rectangle")
            c.edge(task_key, task_key[: -len(".serial")])

        g.edge(request_name, task_key)

    for response_name, task_key in response_data.items():
        g.node(response_name, response_name, shape="oval")

    return g


def visualize(
    tc: "TaskComposition",
    fhandle: BytesIO = None,
    format: str = "png",
    *,
    no_optimization: bool = False,
):
    """Visualize a graph."""
    dsk = tc.pre_optimization_dsk if no_optimization else tc.dsk
    dependencies, dependents = get_deps(dsk)
    g = _dag_to_graphviz(
        dag=dsk,
        dependencies=dependencies,
        request_data=tc.ep_dsk_input_keys,
        response_data=tc.ep_dsk_output_keys,
        no_optimization=no_optimization,
    )
    if fhandle is not None:
        data = g.pipe(format=format)
        fhandle.seek(0)
        fhandle.write(data)
        return

    return g
