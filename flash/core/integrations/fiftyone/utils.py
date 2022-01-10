from itertools import chain
from typing import Dict, List, Optional, Type, Union

import flash
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires

if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    fol = lazy_import("fiftyone.core.labels")
    Label = "fiftyone.Label"
    Session = "fiftyone.Session"
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    fo = None
    fol = None
    Label = object
    Session = object
    SampleCollection = object


@requires("fiftyone")
def visualize(
    predictions: Union[List[Label], List[Dict[str, Label]]],
    filepaths: Optional[List[str]] = None,
    label_field: Optional[str] = "predictions",
    wait: Optional[bool] = False,
    **kwargs,
) -> Optional[Session]:
    """Visualizes predictions from a model with a FiftyOne Output in the
    :ref:`FiftyOne App <fiftyone:fiftyone-app>`.

    This method can be used in all of the following environments:

    -   **Local Python shell**: The App will launch in a new tab in your
        default web browser.
    -   **Remote Python shell**: Pass the ``remote=True`` option to this method
        and then follow the instructions printed to your remote shell to open
        the App in your browser on your local machine.
    -   **Jupyter notebook**: The App will launch in the output of your current
        cell.
    -   **Google Colab**: The App will launch in the output of your current
        cell.
    -   **Python script**: Pass the ``wait=True`` option to block execution of
        your script until the App is closed.

    See :ref:`this page <fiftyone:environments>` for more information about
    using the FiftyOne App in different environments.

    Args:
        predictions: Can be either a list of FiftyOne labels that will be
            matched with the corresponding ``filepaths``, or a list of
            dictionaries with "filepath" and "predictions" keys that contains
            the filepaths and predictions.
        filepaths: A list of filepaths to images or videos corresponding to the
            provided ``predictions``.
        label_field: The name of the label field in which to store the
            predictions in the FiftyOne dataset.
        wait: Whether to block execution until the FiftyOne App is closed.
        **kwargs: Optional keyword arguments for
            :meth:`fiftyone:fiftyone.core.session.launch_app`.

    Returns:
        a :class:`fiftyone:fiftyone.core.session.Session`
    """
    if flash._IS_TESTING:
        return None

    # Flatten list if batches were used
    if all(isinstance(fl, list) for fl in predictions):
        predictions = list(chain.from_iterable(predictions))

    if all(isinstance(fl, dict) for fl in predictions):
        filepaths = [lab["filepath"] for lab in predictions]
        labels = [lab["predictions"] for lab in predictions]
    else:
        labels = predictions

    if filepaths is None:
        raise ValueError("The `filepaths` argument is required if filepaths are not provided in `labels`.")

    dataset = fo.Dataset()
    if filepaths:
        dataset.add_samples([fo.Sample(filepath=f, **{label_field: l}) for f, l in zip(filepaths, labels)])

    session = fo.launch_app(dataset, **kwargs)
    if wait:
        session.wait()

    return session


class FiftyOneLabelUtilities:
    """Helper class providing useful methods for working with ``SampleCollection`` datasets from FiftyOne.

    Args:
        label_field: The field in the ``SampleCollection`` containing the ground truth labels.
        label_cls: The ``FiftyOne.Label`` subclass to expect ground truth labels to be instances of. If ``None``,
            defaults to ``FiftyOne.Label``.
    """

    def __init__(self, label_field: str = "ground_truth", label_cls: Optional[Type[Label]] = None):
        super().__init__()
        self.label_field = label_field
        self.label_cls = label_cls or fol.Label

    def validate(self, sample_collection: SampleCollection):
        label_type = sample_collection._get_label_field_type(self.label_field)
        if not issubclass(label_type, self.label_cls):
            raise ValueError(f"Expected field '{self.label_field}' to have type {self.label_cls}; found {label_type}")

    def get_classes(self, sample_collection: SampleCollection):
        classes = sample_collection.classes.get(self.label_field, None)

        if not classes:
            classes = sample_collection.default_classes

        if not classes:
            label_path = sample_collection._get_label_field_path(self.label_field, "label")[1]
            classes = sample_collection.distinct(label_path)

        return classes
