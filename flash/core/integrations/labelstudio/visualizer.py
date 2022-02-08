import json
import random
import string

from pytorch_lightning.utilities.cloud_io import get_filesystem

from flash.core.data.data_module import DataModule
from flash.core.integrations.labelstudio.input import LabelStudioParameters


class App:
    """App for visualizing predictions in Label Studio results format."""

    def __init__(self, datamodule: DataModule):
        self.parameters: LabelStudioParameters = datamodule.train_dataset.parameters

    def show_predictions(self, predictions):
        """Converts predictions to Label Studio results."""
        results = []
        for prediction_batch in predictions:
            for pred in prediction_batch:
                results.append(self._construct_result(pred))
        return results

    def show_tasks(self, predictions, export_json=None):
        """Converts predictions to tasks format."""
        results = self.show_predictions(predictions)
        data_type = list(self.parameters.data_types)[0]
        meta = {"ids": [], "data": [], "meta": [], "max_predictions_id": 0, "project": None}
        if export_json:
            fs = get_filesystem(export_json)
            with fs.open(export_json) as f:
                _raw_data = json.load(f)
            for task in _raw_data:
                if results:
                    res = results.pop()
                meta["max_predictions_id"] = meta["max_predictions_id"] + 1
                temp = {
                    "result": res["result"],
                    "id": meta["max_predictions_id"],
                    "model_version": "",
                    "score": 0.0,
                    "task": task["id"],
                }
                if task.get("predictions"):
                    task["predictions"].append(temp)
                else:
                    task["predictions"] = [temp]
            return _raw_data
        else:
            print("No export file provided, meta information is generated!")
            final_results = []
            for res in results:
                temp = {
                    "result": [res],
                    "id": meta["max_predictions_id"],
                    "model_version": "",
                    "score": 0.0,
                    "task": meta["max_predictions_id"],
                }
                task = {
                    "id": meta["max_predictions_id"],
                    "predictions": [temp],
                    "data": {data_type: ""},
                    "project": 1,
                }
                meta["max_predictions_id"] = meta["max_predictions_id"] + 1
                final_results.append(task)
            return final_results

    def _construct_result(self, pred):
        """Construction Label Studio result from data source and prediction values."""
        # get label
        if isinstance(pred, list):
            label = [list(self.parameters.classes)[p] for p in pred]
        else:
            label = list(self.parameters.classes)[pred]
        # get data type, if len(data_types) > 1 take first data type
        data_type = list(self.parameters.data_types)[0]
        # get tag type, if len(tag_types) > 1 take first tag
        tag_type = list(self.parameters.tag_types)[0]
        js = {
            "result": [
                {
                    "id": "".join(
                        random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                        for _ in range(10)
                    ),
                    "from_name": "tag",
                    "to_name": data_type,
                    "type": tag_type,
                    "value": {tag_type: label if isinstance(label, list) else [label]},
                }
            ]
        }
        return js


def launch_app(datamodule: DataModule) -> "App":
    """Creating instance of Visualizing App."""
    return App(datamodule)
