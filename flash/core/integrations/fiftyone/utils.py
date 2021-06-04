from time import sleep as sleep_fn
from typing import List, Optional

import flash
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo
    from fiftyone import Sample
    from fiftyone.core.session import Session
else:
    Sample = None
    Session = None


def fiftyone_launch_app(samples: List[Sample], sleep: Optional[int] = 120, **kwargs) -> Optional[Session]:
    if not _FIFTYONE_AVAILABLE:
        raise ModuleNotFoundError("Please, `pip install fiftyone`.")
    if flash._IS_TESTING:
        return None
    dataset = fo.Dataset()
    for sample in samples:
        dataset.add_samples(sample)
    session = fo.launch_app(dataset, **kwargs)
    if sleep:
        sleep_fn(sleep)
    return session


def get_classes(data, label_field: str):
    classes = data.classes.get(label_field, None)

    if not classes:
        classes = data.default_classes

    if not classes:
        label_path = data._get_label_field_path(label_field, "label")[1]
        classes = data.distinct(label_path)

    return classes
