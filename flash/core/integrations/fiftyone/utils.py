from itertools import chain
from typing import Dict, List, Optional, Union

import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo
    from fiftyone.core.labels import Label
    from fiftyone.core.sample import Sample
    from fiftyone.core.session import Session
    from fiftyone.utils.data.parsers import LabeledImageTupleSampleParser
else:
    fo = None
    SampleCollection = None
    Label = None
    Sample = None
    Session = None


def visualize(
    labels: Union[List[Label], List[Dict[str, Label]]],
    filepaths: Optional[List[str]] = None,
    wait: Optional[bool] = True,
    label_field: Optional[str] = "predictions",
    **kwargs
) -> Optional[Session]:
    """Use the result of a FiftyOne serializer to visualize predictions in the
    FiftyOne App.

    Args:
        labels: Either a list of FiftyOne labels that will be applied to the
            corresponding filepaths provided with through `filepath` or
            `datamodule`. Or a list of dictionaries containing image/video
            filepaths and corresponding FiftyOne labels.
        filepaths: A list of filepaths to images or videos corresponding to the
            provided `labels`.
        wait: A boolean determining whether to launch the FiftyOne session and
            wait until the session is closed or whether to return immediately.
        label_field: The string of the label field in the FiftyOne dataset
            containing predictions
    """
    if not _FIFTYONE_AVAILABLE:
        raise ModuleNotFoundError("Please, `pip install fiftyone`.")
    if flash._IS_TESTING:
        return None

    # Flatten list if batches were used
    if all(isinstance(fl, list) for fl in labels):
        labels = list(chain.from_iterable(labels))

    if all(isinstance(fl, dict) for fl in labels):
        filepaths = [lab["filepath"] for lab in labels]
        labels = [lab["predictions"] for lab in labels]

    if filepaths is None:
        raise ValueError("The `filepaths` argument is required if filepaths are not provided in `labels`.")

    dataset = fo.Dataset()
    if filepaths:
        dataset.add_labeled_images(
            list(zip(filepaths, labels)),
            LabeledImageTupleSampleParser(),
            label_field=label_field,
        )
    session = fo.launch_app(dataset, **kwargs)
    if wait:
        session.wait()
    return session


def get_classes(data, label_field: str):
    classes = data.classes.get(label_field, None)

    if not classes:
        classes = data.default_classes

    if not classes:
        label_path = data._get_label_field_path(label_field, "label")[1]
        classes = data.distinct(label_path)

    return classes
