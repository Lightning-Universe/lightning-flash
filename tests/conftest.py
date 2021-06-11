import os
import pathlib
import shutil
from typing import Tuple, TYPE_CHECKING

import pytest
import torch
import torchvision
from pytest_mock import MockerFixture

from flash.core.serve.decorators import uuid4  # noqa (used in mocker.patch)

if TYPE_CHECKING:
    from flash.core.serve import GridModel


class UUID_String(str):
    """Class to replace UUID object with str instance and hex attribute"""

    @property
    def hex(self):
        return str(self)


@pytest.fixture(scope="function", autouse=True)
def patch_decorators_uuid_generator_func(mocker: MockerFixture):
    call_num = 0

    def _generate_sequential_uuid():
        nonlocal call_num
        call_num += 1
        return UUID_String(f"callnum_{call_num}")

    mocker.patch("flash.core.serve.decorators.uuid4", side_effect=_generate_sequential_uuid)
    yield


@pytest.fixture(scope="session")
def original_global_datadir():
    return pathlib.Path(os.path.realpath(__file__)).parent.joinpath("serve").joinpath("data")


def prep_global_datadir(tmp_path_factory, original_global_datadir):
    temp_dir = tmp_path_factory.mktemp("data") / "datadir"
    shutil.copytree(original_global_datadir, temp_dir)
    return temp_dir


@pytest.fixture(scope="session")
def session_global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


@pytest.fixture(scope="module")
def module_global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


@pytest.fixture(scope="function")
def global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


@pytest.fixture(scope="session")
def squeezenet1_1_model():
    model = torchvision.models.squeezenet1_1(pretrained=True).eval()
    yield model


@pytest.fixture(scope="session")
def lightning_squeezenet1_1_obj():
    from tests.serve.models import LightningSqueezenet

    model = LightningSqueezenet()
    model.eval()
    yield model


@pytest.fixture(scope="session")
def squeezenet_gridmodel(squeezenet1_1_model, session_global_datadir) -> Tuple[GridModel, str]:
    from flash.core.serve import GridModel

    trace = torch.jit.trace(squeezenet1_1_model.eval(), (torch.rand(1, 3, 224, 224), ))
    fpth = str(session_global_datadir / "squeezenet_jit_trace.pt")
    torch.jit.save(trace, fpth)

    model = GridModel(fpth)
    yield (model, fpth)


@pytest.fixture()
def lightning_squeezenet_checkpoint_path(tmp_path):
    from tests.serve.models import LightningSqueezenet

    model = LightningSqueezenet()
    state_dict = {"state_dict": model.state_dict()}
    path = tmp_path / "model.pth"
    torch.save(state_dict, path)
    yield path
    path.unlink()
