import base64
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from flash.core.serve.dag.task import get
from flash.core.serve.dag.visualize import visualize
from flash.core.serve.execution import (
    build_composition,
    component_dag_content,
    ComponentJSON,
    merged_dag_content,
    MergedJSON,
    TaskComposition,
)
from flash.core.serve.interfaces.models import Alive, EndpointProtocol
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _FASTAPI_AVAILABLE

if _CYTOOLZ_AVAILABLE:
    from cytoolz import first
else:
    first = None

if _FASTAPI_AVAILABLE:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
else:
    FastAPI, Request, HTMLResponse, Jinja2Templates = object, object, object, object

if TYPE_CHECKING:  # pragma: no cover
    from flash.core.serve.component import ModelComponent
    from flash.core.serve.composition import Composition

try:
    from typing import ForwardRef

    RequestModel = ForwardRef("RequestModel")
    ResponseModel = ForwardRef("ResponseModel")
except ImportError:
    RequestModel = None
    ResponseModel = None


def _build_endpoint(
    request_model: RequestModel,
    dsk_composition: TaskComposition,
    response_model: ResponseModel,
) -> Callable[[RequestModel], ResponseModel]:
    def endpoint_fn(body: request_model):
        session = body.session if body.session else str(uuid.uuid4())
        _res = get(
            dsk_composition.dsk,
            dsk_composition.get_keys,
            cache=body.payload.dict(),
            sortkeys=dsk_composition.sortkeys,
        )
        return {
            "result": dict(zip(dsk_composition.ep_dsk_output_keys, _res)),
            "session": session,
        }

    endpoint_fn.__globals__["request_model"] = request_model
    endpoint_fn.__globals__["response_model"] = response_model
    return endpoint_fn


def _build_meta(Body: RequestModel) -> Callable[[], Dict[str, Any]]:
    def meta() -> Dict[str, Any]:
        nonlocal Body
        return Body.schema()

    return meta


def _build_alive_check() -> Callable[[], Alive]:
    def alive() -> Alive:
        return Alive.construct(alive=True)

    return alive


def _build_visualization(
    dsk_composition: TaskComposition,
    templates: Jinja2Templates,
    *,
    no_optimization: bool = False,
):
    def endpoint_visualization(request: Request):
        nonlocal dsk_composition, templates, no_optimization
        with BytesIO() as f:
            visualize(dsk_composition, fhandle=f, no_optimization=no_optimization)
            f.seek(0)
            raw = f.read()
        encoded = base64.b64encode(raw).decode("ascii")
        res = templates.TemplateResponse("dag.html", {"request": request, "encoded_image": encoded})
        return res

    return endpoint_visualization


def _build_dag_json(
    components: Dict[str, "ModelComponent"],
    ep_proto: Optional["EndpointProtocol"],
    *,
    show_connected_components: bool = True,
):
    if show_connected_components is True:

        def dag_json():
            return merged_dag_content(ep_proto, components).dict()

    else:

        def dag_json():
            return component_dag_content(components).dict()

    return dag_json


def setup_http_app(composition: "Composition", debug: bool) -> "FastAPI":
    from flash import __version__

    app = FastAPI(
        debug=debug,
        version=__version__,
        title="FlashServe",
    )
    # Endpoint Route
    #   `/flashserve/alive`
    app.get(
        "/flashserve/alive",
        name="alive",
        description="If you can reach this endpoint, the server is runnning.",
        response_model=Alive,
    )(_build_alive_check())

    _no_optimization_dsk = build_composition(
        endpoint_protocol=first(composition.endpoint_protocols.values()),
        components=composition.components,
        connections=composition.connections,
    )
    pth = Path(__file__).parent.joinpath("templates")
    templates = Jinja2Templates(directory=str(pth.absolute()))

    # Endpoint Route
    #   `/flashserve/component_dags`
    app.get(
        "/flashserve/component_dags",
        name="component_dags",
        summary="HTML Rendering of Component DAGs",
        response_class=HTMLResponse,
    )(_build_visualization(dsk_composition=_no_optimization_dsk, templates=templates, no_optimization=True))

    # Endpoint Route
    #   `/flashserve/dag_json`
    app.get(
        "/flashserve/dag_json",
        name="components JSON DAG",
        summary="JSON representation of component DAG",
        response_model=ComponentJSON,
    )(
        _build_dag_json(
            components=composition.components,
            ep_proto=None,
            show_connected_components=False,
        )
    )

    for ep_name, ep_proto in composition.endpoint_protocols.items():
        dsk = build_composition(
            endpoint_protocol=ep_proto,
            components=composition.components,
            connections=composition.connections,
        )
        RequestModel = ep_proto.request_model  # skipcq: PYL-W0621
        ResponseModel = ep_proto.response_model  # skipcq: PYL-W0621

        # Endpoint Route
        #   `/{proto}
        app.post(
            f"{ep_proto.route}",
            name=ep_name,
            tags=[ep_name],
            summary="Perform a Compution.",
            description="Computes results of DAG defined by these components & endpoint.",
            response_model=ResponseModel,
        )(_build_endpoint(RequestModel, dsk, ResponseModel))

        # Endpoint Route:
        #   `/{proto}/meta`
        app.get(
            f"{ep_proto.route}/meta",
            name=f"{ep_name} meta schema",
            tags=[ep_name],
            summary="OpenAPI schema",
            description="OpenAPI schema for this endpoints's compute route.",
        )(_build_meta(RequestModel))

        # Endpoint Route
        #   `/{proto}/dag`
        app.get(
            f"{ep_proto.route}/dag",
            name=f"{ep_name} DAG Visualization",
            tags=[ep_name],
            summary="HTML Rendering of DAG",
            description=(
                "Displays an html image rendering the DAG of functions "
                "& components executed to reach the endpoint outputs."
            ),
            response_class=HTMLResponse,
        )(_build_visualization(dsk, templates))

        # Endpoint Route
        #   `/{proto}/dag_json`
        app.get(
            f"{ep_proto.route}/dag_json",
            name=f"{ep_name} JSON DAG",
            tags=[ep_name],
            summary="JSON representatino of DAG",
            response_model=MergedJSON,
        )(
            _build_dag_json(
                components=composition.components,
                ep_proto=ep_proto,
                show_connected_components=True,
            )
        )
    return app
