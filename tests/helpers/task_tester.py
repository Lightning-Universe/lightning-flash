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
import inspect
import os
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import pytest
import torch

from flash.__main__ import main
from flash.core.model import Task


def _test_forward(self):
    """Tests that ``Task.forward`` applied to the example input gives the expected output."""
    model = self.instantiated_task
    output = model(self.example_forward_input)
    self.check_forward_output(output)


def _test_jit_trace(self, tmpdir):
    """Tests that the task can be traced and saved with JIT then reloaded and used."""
    path = os.path.join(tmpdir, "test.pt")

    model = self.instantiated_task
    model.eval()

    model = torch.jit.trace(model, self.example_forward_input)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    self.check_forward_output(model(self.example_forward_input))


def _test_jit_script(self, tmpdir):
    """Tests that the task can be scripted and saved with JIT then reloaded and used."""
    path = os.path.join(tmpdir, "test.pt")

    model = self.instantiated_task
    model.eval()

    model = torch.jit.script(model)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    self.check_forward_output(model(self.example_forward_input))


def _test_cli(self):
    """Tests that the default Flash zero configuration runs for the task."""
    cli_args = ["flash", self.cli_command, "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass


def _test_load_from_checkpoint_dependency_error(self):
    """Tests that a ``ModuleNotFoundError`` is raised when ``load_from_checkpoint`` is called if the required
    dependencies are not available."""
    with pytest.raises(ModuleNotFoundError, match=re.escape("Required dependencies not available.")):
        self.task.load_from_checkpoint("not_a_real_checkpoint.pt")


class TaskTesterMeta(ABCMeta):
    """The ``TaskTesterMeta`` is a metaclass which attaches a suite of tests to classes that extend ``TaskTester``
    based on the configuration variables they define.

    These tests will also be wrapped with the appropriate marks to skip them if the required dependencies are not
    available.
    """

    def __new__(mcs, *args, **kwargs):
        result = ABCMeta.__new__(mcs, *args, **kwargs)

        # Attach forward test
        result.test_forward = _test_forward

        # Attach JIT tests
        if result.traceable:
            result.test_jit_trace = _test_jit_trace

        if result.scriptable:
            result.test_jit_script = _test_jit_script

        # Attach CLI test
        if result.cli_command is not None:
            result.test_cli = _test_cli

        # Skip tests if dependencies not available
        regex = "( test_* )"
        for attribute_name, attribute_value in filter(lambda x: re.match(regex, x[0]), inspect.getmembers(result)):
            setattr(
                result,
                attribute_name,
                pytest.mark.skipif(not result.dependencies_available, reason="Dependencies not available.")(
                    attribute_value
                ),
            )

        # Attach error check test
        result.test_load_from_checkpoint_dependency_error = pytest.mark.skipif(
            result.dependencies_available, reason="Dependencies available."
        )(_test_load_from_checkpoint_dependency_error)

        return result


class TaskTester(metaclass=TaskTesterMeta):
    """The ``TaskTester`` should be extended to automatically run a suite of tests for each ``Task``.

    Use the class attributes to control which tests will be run. For example, if ``traceable`` is ``False`` then no JIT
    tracing test will be performed.
    """

    task: Task
    task_args: Tuple = ()
    task_kwargs: Dict = {}
    cli_command: Optional[str] = None
    dependencies_available: bool = True
    traceable: bool = True
    scriptable: bool = True

    @property
    def instantiated_task(self):
        return self.task(*self.task_args, **self.task_kwargs)

    @abstractmethod
    @property
    def example_forward_input(self):
        pass

    @abstractmethod
    def check_forward_output(self, output: Any):
        """Override this hook to check the output of ``Task.forward`` with random data of the required shape."""
        pass
