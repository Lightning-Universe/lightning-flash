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
import functools
import inspect
import os
import types
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import pytest
import torch
from torch.utils.data import Dataset

import flash
from flash.__main__ import main
from flash.core.model import Task


def _copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


class StaticDataset(Dataset):
    def __init__(self, sample, length):
        super().__init__()

        self.sample = sample
        self.length = length

    def __getitem__(self, _):
        return self.sample

    def __len__(self):
        return self.length


def _test_forward(self):
    """Tests that ``Task.forward`` applied to the example input gives the expected output."""
    model = self.instantiated_task
    model.eval()
    output = model(self.example_forward_input)
    self.check_forward_output(output)


def _test_fit(self, tmpdir, task_kwargs):
    """Tests that a single batch fit pass completes."""
    dataset = StaticDataset(self.example_train_sample, 4)

    args = self.task_args
    kwargs = dict(**self.task_kwargs)
    kwargs.update(task_kwargs)
    model = self.task(*args, **kwargs)

    trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, model.process_train_dataset(dataset, batch_size=4))


def _test_val(self, tmpdir, task_kwargs):
    """Tests that a single batch validation pass completes."""
    dataset = StaticDataset(self.example_val_sample, 4)

    args = self.task_args
    kwargs = dict(**self.task_kwargs)
    kwargs.update(task_kwargs)
    model = self.task(*args, **kwargs)

    trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.validate(model, model.process_val_dataset(dataset, batch_size=4))


def _test_test(self, tmpdir, task_kwargs):
    """Tests that a single batch test pass completes."""
    dataset = StaticDataset(self.example_test_sample, 4)

    args = self.task_args
    kwargs = dict(**self.task_kwargs)
    kwargs.update(task_kwargs)
    model = self.task(*args, **kwargs)

    trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, model.process_test_dataset(dataset, batch_size=4))


def _test_jit_trace(self, tmpdir):
    """Tests that the task can be traced and saved with JIT then reloaded and used."""
    path = os.path.join(tmpdir, "testing_model.pt")

    model = self.instantiated_task
    model.eval()

    model = model.to_torchscript(method="trace", example_inputs=self.example_forward_input)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    self.check_forward_output(model(self.example_forward_input))


def _test_jit_script(self, tmpdir):
    """Tests that the task can be scripted and saved with JIT then reloaded and used."""
    path = os.path.join(tmpdir, "testing_model.pt")

    model = self.instantiated_task
    model.eval()

    model = torch.jit.script(model)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    self.check_forward_output(model(self.example_forward_input))


def _test_cli(self, extra_args: List):
    """Tests that the default Flash zero configuration runs for the task."""
    cli_args = ["flash", self.cli_command, "--trainer.fast_dev_run", "True"] + extra_args
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass


def _test_load_from_checkpoint_dependency_error(self):
    """Tests that a ``ModuleNotFoundError`` is raised when ``load_from_checkpoint`` is called if the required
    dependencies are not available."""
    with pytest.raises(ModuleNotFoundError, match="Required dependencies not available."):
        self.task.load_from_checkpoint("not_a_real_checkpoint.pt")


def _test_init_dependency_error(self):
    """Tests that a ``ModuleNotFoundError`` is raised when the task is instantiated if the required dependencies
    are not available."""
    with pytest.raises(ModuleNotFoundError, match="Required dependencies not available."):
        _ = self.instantiated_task


class TaskTesterMeta(ABCMeta):
    """The ``TaskTesterMeta`` is a metaclass which attaches a suite of tests to classes that extend ``TaskTester``
    based on the configuration variables they define.

    These tests will also be wrapped with the appropriate marks to skip them if the required dependencies are not
    available.
    """

    @staticmethod
    def attach_test(task_tester, test_name, test):
        test = _copy_func(test)

        # Resolve marks
        marks = task_tester.marks.get(test_name, None)
        if marks is not None:
            if not isinstance(marks, List):
                marks = [marks]

            for mark in marks:
                test = mark(test)

        # Attach test
        setattr(task_tester, test_name, test)

    def __new__(mcs, name: str, bases: Tuple, class_dict: Dict[str, Any]):
        result = ABCMeta.__new__(mcs, name, bases, class_dict)

        # Skip attaching for the base class
        if name == "TaskTester":
            return result

        # Attach forward test
        if "example_forward_input" in class_dict:
            mcs.attach_test(result, "test_forward", _test_forward)

        # Attach fit test
        if "example_train_sample" in class_dict:
            mcs.attach_test(result, "test_fit", _test_fit)

        # Attach val test
        if "example_val_sample" in class_dict:
            mcs.attach_test(result, "test_val", _test_val)

        # Attach test test
        if "example_test_sample" in class_dict:
            mcs.attach_test(result, "test_test", _test_test)

        # Attach JIT tests
        if result.traceable and "example_forward_input" in class_dict:
            mcs.attach_test(result, "test_jit_trace", _test_jit_trace)

        if result.scriptable and "example_forward_input" in class_dict:
            mcs.attach_test(result, "test_jit_script", _test_jit_script)

        # Attach CLI test
        if result.cli_command is not None:
            mcs.attach_test(result, "test_cli", _test_cli)

        # Skip tests if dependencies not available
        for attribute_name, attribute_value in filter(lambda x: x[0].startswith("test"), inspect.getmembers(result)):
            setattr(
                result,
                attribute_name,
                pytest.mark.skipif(not result.is_testing, reason="Dependencies not available.")(
                    _copy_func(attribute_value)
                ),
            )

        # Attach error check tests
        mcs.attach_test(
            result, "test_load_from_checkpoint_dependency_error", _test_load_from_checkpoint_dependency_error
        )

        mcs.attach_test(result, "test_init_dependency_error", _test_init_dependency_error)

        for dependency_test in ["test_load_from_checkpoint_dependency_error", "test_init_dependency_error"]:
            setattr(
                result,
                dependency_test,
                pytest.mark.skipif(result.is_available, reason="Dependencies available.")(
                    _copy_func(getattr(result, dependency_test))
                ),
            )

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
    traceable: bool = True
    scriptable: bool = True
    is_available: bool = True
    is_testing: bool = True

    marks: Dict[str, Any] = {
        "test_fit": [pytest.mark.parametrize("task_kwargs", [{}])],
        "test_val": [pytest.mark.parametrize("task_kwargs", [{}])],
        "test_test": [pytest.mark.parametrize("task_kwargs", [{}])],
        "test_cli": [pytest.mark.parametrize("extra_args", [[]])],
    }

    trainer_args: Tuple = ()
    trainer_kwargs: Dict = {}

    @property
    def instantiated_task(self):
        return self.task(*self.task_args, **self.task_kwargs)

    @property
    def instantiated_trainer(self):
        return flash.Trainer(*self.trainer_args, **self.trainer_kwargs)

    @property
    def example_forward_input(self):
        pass

    def check_forward_output(self, output: Any):
        """Override this hook to check the output of ``Task.forward`` with random data of the required shape."""
        pass

    @property
    def example_train_sample(self):
        pass

    @property
    def example_val_sample(self):
        pass

    @property
    def example_test_sample(self):
        pass
