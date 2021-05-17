import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE
from flash.graph.classification.data import GraphClassificationData
from flash.graph.classification.model import GraphClassifier

if _PYTORCH_GEOMETRIC_AVAILABLE:
    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset
else:
    raise ModuleNotFoundError("Please, pip install -e '.[graph]'")

dataset = TUDataset("data", name='IMDB-BINARY')
num_features = 136
transform = {
    "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.OneHotDegree(num_features - 1)),
    "to_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.ToSparseTensor())
}
dm = GraphClassificationData.from_datasets(
    train_dataset=dataset[:len(dataset) // 2],
    test_dataset=dataset[len(dataset) // 2:],
    val_split=0.1,
    train_transform=transform,
    val_transform=transform,
    predict_transform=transform,
    num_features=num_features,
)

model = GraphClassifier(num_features=num_features, num_classes=dm.num_classes)

trainer = flash.Trainer(max_epochs=1)
trainer.fit(model, datamodule=dm)

# 7. Save it!
trainer.save_checkpoint("graph_classification.pt")
