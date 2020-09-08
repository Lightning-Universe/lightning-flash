import pytest
from copy import deepcopy
from pytorch_lightning.core.step_result import EvalResult, TrainResult
from torch.optim import Adam, SGD

from pl_flash import Flash, Trainer
from pytorch_lightning.metrics import functional as FM

import torch
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.data import DataLoader


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, shape: tuple, num_classes: int, length: int):

        super().__init__()

        self.shape = shape
        self.num_classes = num_classes
        self.length = length

    def __getitem__(self, index: int):
        return torch.rand(self.shape), torch.randint(self.num_classes, size=(1,)).item()

    def __len__(self):
        return self.length


def instantiate_and_train(tmpdir, loss, metrics, optimizer_cls):
    data = DataLoader(DummyDataset((1, 28, 28), 10, 500), batch_size=64, shuffle=True)
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())

    model = Flash(mlp, loss=loss, metrics=metrics, optimizer=optimizer_cls)

    assert isinstance(model.model, torch.nn.Sequential)
    assert isinstance(model.losses, (dict, torch.nn.ModuleDict))
    assert isinstance(model.metrics, (dict, torch.nn.ModuleDict))

    if isinstance(optimizer_cls, str):
        optimizer_cls = getattr(torch.optim, optimizer_cls)
    assert isinstance(model.configure_optimizers(), optimizer_cls)

    Trainer(fast_dev_run=True, default_root_dir=tmpdir, max_steps=2).fit(model, data)


@pytest.mark.parametrize(
    "loss",
    [
        torch.nn.NLLLoss(),
        F.nll_loss,
        {"loss": F.nll_loss},
        {"loss": torch.nn.NLLLoss()},
        [torch.nn.NLLLoss()],
        [F.nll_loss],
        "nll_loss",
        "NLLLoss",
        ["NLLLoss"],
        ["nll_loss"],
        {"loss": "NLLLoss"},
        {"loss": "nll_loss"},
    ],
)
def test_loss_argument_types(tmpdir, loss):
    return instantiate_and_train(tmpdir, loss, None, Adam)


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        torch.nn.NLLLoss(),
        F.nll_loss,
        {"loss": F.nll_loss},
        {"loss": torch.nn.NLLLoss()},
        [torch.nn.NLLLoss()],
        [F.nll_loss],
        "nll_loss",
        "NLLLoss",
        ["NLLLoss"],
        ["nll_loss"],
        {"loss": "NLLLoss"},
        {"loss": "nll_loss"},
    ],
)
def test_metrics_argument_types(tmpdir, metrics):
    return instantiate_and_train(tmpdir, F.nll_loss, metrics, Adam)


@pytest.mark.parametrize("optimizer_cls", [Adam, SGD, "Adam", "SGD"])
def test_optimizers_argument_types(tmpdir, optimizer_cls):
    return instantiate_and_train(tmpdir, F.nll_loss, None, optimizer_cls)


def test_forward():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    model = Flash(mlp, loss=F.nll_loss)

    rand_inp = torch.rand(10, 1, 28 * 28)
    assert torch.allclose(model(rand_inp), mlp(rand_inp))


def test_configure_optimizer():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    model = Flash(mlp, loss=F.nll_loss, optimizer=Adam)

    opt = model.configure_optimizers()
    assert isinstance(opt, Adam)


def test_compute_dict():
    expected_vals = {"a": torch.tensor(1), "b": 2}
    metrics = {"a": lambda y, y_hat: torch.tensor(1), "b": lambda y, y_hat: 2}
    metric_vals = Flash.compute_dict(metrics, y=torch.zeros(1), y_hat=torch.zeros(1), prefix="", sep="")

    assert isinstance(metric_vals, dict)
    assert len(metric_vals) == len(expected_vals)
    for k, v in metric_vals.items():
        assert k in expected_vals
        assert v == expected_vals[k]


def test_compute_metrics():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    expected_vals = {"a": torch.tensor(1), "b": 2}
    model = Flash(mlp, loss=F.nll_loss, metrics={"a": lambda y, y_hat: torch.tensor(1), "b": lambda y, y_hat: 2})
    metric_vals = model.compute_metrics(y=torch.zeros(1), y_hat=torch.zeros(1), prefix="", sep="")

    assert isinstance(metric_vals, dict)
    assert len(metric_vals) == len(expected_vals)
    for k, v in metric_vals.items():
        assert k in expected_vals
        assert v == expected_vals[k]


def test_compute_loss():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    expected_vals = {"a": torch.tensor(1), "b": torch.tensor(2)}
    model = Flash(mlp, loss={"a": lambda y, y_hat: torch.tensor(1), "b": lambda y, y_hat: torch.tensor(2)})
    total_loss, loss_vals = model.compute_loss(y=torch.zeros(1), y_hat=torch.zeros(1), prefix="", sep="")

    assert isinstance(loss_vals, dict)
    assert isinstance(total_loss, torch.Tensor)
    assert len(loss_vals) == len(expected_vals)
    for k, v in loss_vals.items():
        assert k in expected_vals
        assert v == expected_vals[k]

    total_loss_recalc = 0

    for v in expected_vals.values():
        total_loss_recalc += v

    assert torch.allclose(total_loss, total_loss_recalc)


def test_train_step():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    expected_keys = ["a", "b", "c", "d"]
    model = Flash(
        mlp,
        loss={"a": torch.nn.NLLLoss(), "b": torch.nn.NLLLoss()},
        metrics={"c": lambda y, y_hat: 3, "d": lambda y, y_hat: 4},
    )

    model.train()

    batch = torch.rand(10, 1, 28, 28), torch.randint(10, size=(10,))
    result = model.training_step(batch, batch_idx=0)

    assert isinstance(result, TrainResult)
    for k in expected_keys:
        assert f"train/{k}" in result

    assert isinstance(result["minimize"], torch.Tensor)
    assert result["minimize"].grad_fn is not None


def _val_test_step(prefix: str):
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax())
    expected_keys = ["a", "b", "c", "d"]
    model = Flash(
        mlp,
        loss={"a": torch.nn.NLLLoss(), "b": torch.nn.NLLLoss()},
        metrics={"c": lambda y, y_hat: 3, "d": lambda y, y_hat: 4},
    )

    batch = torch.rand(10, 1, 28, 28), torch.randint(10, size=(10,))

    if prefix == "val":
        result = model.validation_step(batch, batch_idx=0)
    else:
        result = model.test_step(batch, batch_idx=0)

    assert isinstance(result, EvalResult)
    for k in expected_keys:
        assert f"{prefix}/{k}" in result

    with pytest.raises(KeyError):
        result["minimize"]


def test_val_step():
    return _val_test_step("val")


def test_test_step():
    return _val_test_step("test")


@pytest.mark.parametrize(
    ["batch", "x", "y"],
    [
        pytest.param([1.0, 2.0], 1.0, 2.0),
        pytest.param({"x": 1.0, "y": 2}, 1.0, 2.0),
        pytest.param(torch.tensor(1.0), torch.tensor(1.0), None),
    ],
)
def test_unpack_data(batch, x, y):
    model = Flash(torch.nn.Linear(1, 1), F.nll_loss)
    x_unpacked, y_unpacked = model.unpack_batch(batch)

    assert x == x_unpacked

    if y is None:
        assert y_unpacked is None

    else:
        assert y == y_unpacked
