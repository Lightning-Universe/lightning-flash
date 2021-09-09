import random
import string

from flash.core.data.data_module import DataModule


class App:
    """App for visualizing predictions in Label Studio results format."""
    def __init__(self, datamodule: DataModule):
        self.datamodule = datamodule

    def show_predictions(self, predictions):
        """Converts predictions to Label Studio results."""
        results = []
        for pred in predictions:
            results.append(self.construct_result(pred))
        return results

    def construct_result(self, pred):
        """Construction Label Studio result from data source and prediction values."""
        ds = self.datamodule.data_source
        # get label
        if isinstance(pred, list):
            label = [list(ds.classes)[p] for p in pred]
        else:
            label = list(ds.classes)[pred]
        # get data type, if len(data_types) > 1 take first data type
        data_type = list(ds.data_types)[0]
        # get tag type, if len(tag_types) > 1 take first tag
        tag_type = list(ds.tag_types)[0]
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
    return App(datamodule)
