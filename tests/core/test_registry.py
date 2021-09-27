# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from flash.core.registry import ConcatRegistry, ExternalRegistry, FlashRegistry


def test_registry_raises():
    backbones = FlashRegistry("backbones")

    @backbones
    def my_model(nc_input=5, nc_output=6):
        return nn.Linear(nc_input, nc_output), nc_input, nc_output

    with pytest.raises(MisconfigurationException, match="You can only register a callable, found: 3"):
        backbones(3, name="foo")

    backbones(my_model, name="foo", override=True)

    with pytest.raises(MisconfigurationException, match="Function with name: foo and metadata: {}"):
        backbones(my_model, name="foo", override=False)

    with pytest.raises(KeyError, match="Found no matches"):
        backbones.get("foo", baz="bar")

    backbones.remove("foo")
    with pytest.raises(KeyError, match="Key: foo is not in FlashRegistry"):
        backbones.get("foo")

    with pytest.raises(TypeError, match="name` must be a str"):
        backbones(name=float)  # noqa


def test_registry():
    backbones = FlashRegistry("backbones")

    @backbones
    def my_model(nc_input=5, nc_output=6):
        return nn.Linear(nc_input, nc_output), nc_input, nc_output

    mlp, nc_input, nc_output = backbones.get("my_model")(nc_output=7)
    assert nc_input == 5
    assert nc_output == 7
    assert mlp.weight.shape == (7, 5)

    # basic get
    backbones(my_model, name="foo")
    assert backbones.get("foo")

    # test override
    backbones(my_model, name="foo", override=True)
    functions = backbones.get("foo", strict=False)
    assert len(functions) == 1

    # test metadata filtering
    backbones(my_model, name="foo", namespace="timm", type="resnet")
    backbones(my_model, name="foo", namespace="torchvision", type="resnet")
    backbones(my_model, name="foo", namespace="timm", type="densenet")
    backbones(my_model, name="foo", namespace="timm", type="alexnet")
    function = backbones.get("foo", with_metadata=True, type="resnet", namespace="timm")
    assert function["name"] == "foo"
    assert function["metadata"] == {"namespace": "timm", "type": "resnet"}

    # test strict=False and with_metadata=False
    functions = backbones.get("foo", namespace="timm", strict=False)
    assert len(functions) == 3
    assert all(callable(f) for f in functions)

    # test available keys
    assert backbones.available_keys() == ["foo", "foo", "foo", "foo", "foo", "my_model"]


# todo (tchaton) Debug this test.
@pytest.mark.skipif(True, reason="need investigation")
def test_registry_multiple_decorators(caplog):
    backbones = FlashRegistry("backbones", verbose=True)

    with caplog.at_level(logging.INFO):

        @backbones
        @backbones(name="foo")
        @backbones(name="bar", foobar=True)
        def my_model():
            return 1

    assert caplog.messages == [
        "Registering: my_model function with name: bar and metadata: {'foobar': True}",
        "Registering: my_model function with name: foo and metadata: {}",
        "Registering: my_model function with name: my_model and metadata: {}",
    ]

    assert len(backbones) == 3
    assert "foo" in backbones
    assert "my_model" in backbones
    assert "bar" in backbones


def test_external_registry():
    def getter(key: str):
        return key

    registry = ExternalRegistry(getter, "backbones", "test_provider")
    assert registry.get("testing")() == "testing"
    available = registry.available_keys()
    assert len(available) == 1
    assert "test_provider" in available[0]

    registry = ExternalRegistry(getter, "backbones", ["test_provider_1", "test_provider_2"])
    assert "test_provider_1, test_provider_2" in registry.available_keys()[0]

    registry = ExternalRegistry(getter, "backbones")
    assert len(registry.available_keys()) == 0


def test_concat_registry():
    registry_1 = FlashRegistry("backbones")
    registry_2 = FlashRegistry("backbones")
    registry_3 = FlashRegistry("test")

    @registry_1(name="foo")
    @registry_2(name="foo")
    @registry_2(name="bar")
    @registry_3(name="baz")
    def my_model():
        return 1

    registry = registry_1 + registry_2

    assert isinstance(registry, ConcatRegistry)
    assert "foo" in registry
    assert registry.name == "backbones"
    assert len(registry) == 3
    assert all(not isinstance(r, ConcatRegistry) for r in registry.registries)
    assert len(registry.get("foo", strict=False)) == 2

    registry.remove("foo")
    assert len(registry) == 1
    assert registry.available_keys() == ["bar"]

    registry(my_model)
    assert "my_model" in registry

    new_registry = registry + registry_3
    assert all(not isinstance(r, ConcatRegistry) for r in new_registry.registries)
    assert "baz" in new_registry

    new_registry = registry_3 + registry
    assert all(not isinstance(r, ConcatRegistry) for r in new_registry.registries)
    assert "baz" in new_registry
