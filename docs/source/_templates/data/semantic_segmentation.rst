{% extends "base.rst" %}
{% block from_folders %}
Construct the {{ data_module }} from folders.

{% if data_sources['folders'].extensions is defined %}
The supported file extensions are: {{ data_sources['folders'].extensions|join(', ') }}.
{% set extension = data_sources['folders'].extensions[0] %}
{% else %}
{% set extension = '' %}
{% endif %}

For train, test, and val data, we expect a folder containing inputs and another folder containing the masks.
Here's the required structure:

.. code-block::

    train_folder
    ├── inputs
    │   ├── file1{{ extension }}
    │   ├── file2{{ extension }}
    │   ...
    └── masks
        ├── file1{{ extension }}
        ├── file2{{ extension }}
        ...

For prediction, the folder is expected to contain the files for inference, like this:

.. code-block::

    predict_folder
    ├── file1{{ extension }}
    ├── file2{{ extension }}
    ...

Example::

    data_module = {{ data_module_raw }}.from_folders(
        train_folder = "./train_folder/inputs",
        train_target_folder = "./train_folder/masks",
        predict_folder = "./predict_folder",
        ...
    )
{% endblock %}
{% block from_files %}
Construct the {{ data_module }} from lists of input images and corresponding list of target images.

{% if data_sources['files'].extensions is defined %}
The supported file extensions are: {{ data_sources['files'].extensions|join(', ') }}.
{% set extension = data_sources['files'].extensions[0] %}
{% else %}
{% set extension = '' %}
{% endif %}

Example::

    train_files = ["file1{{ extension }}", "file2{{ extension }}", "file3{{ extension }}", ...]
    train_targets = ["mask1{{ extension }}", "mask2{{ extension }}", "mask3{{ extension }}", ...]

    datamodule = {{ data_module_raw }}.from_files(
        train_files = train_files,
        train_targets = train_targets,
        ...
    )
{% endblock %}
{% block from_datasets %}
{{ super() }}

.. note::

    The ``__getitem__`` of your datasets should return a dictionary with ``"input"`` and ``"target"`` keys which map to the input and target images as tensors.
{% endblock %}
