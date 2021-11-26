{% macro render_subsection(title) -%}
{{ title }}
{{ '_' * title|length }}

{{ caller() }}
{%- endmacro %}

This section details the available ways to load your own data into the {{ data_module }}.

{% if 'folders' in inputs %}
{% call render_subsection('from_folders') %}

{% block from_folders %}
Construct the {{ data_module }} from folders.

{% if inputs['folders'].extensions is defined %}
The supported file extensions are: {{ inputs['folders'].extensions|join(', ') }}.
{% set extension = inputs['folders'].extensions[0] %}
{% else %}
{% set extension = '' %}
{% endif %}

For train, test, and val data, the folders are expected to contain a sub-folder for each class.
Here's the required structure:

.. code-block::

    train_folder
    ├── class_1
    │   ├── file1{{ extension }}
    │   ├── file2{{ extension }}
    │   ...
    └── class_2
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
        train_folder = "./train_folder",
        predict_folder = "./predict_folder",
        ...
    )
{% endblock %}
{% endcall %}
{% endif %}
{% if 'files' in inputs %}
{% call render_subsection('from_files') %}

{% block from_files %}
Construct the {{ data_module }} from lists of files and corresponding lists of targets.

{% if inputs['files'].extensions is defined %}
The supported file extensions are: {{ inputs['files'].extensions|join(', ') }}.
{% set extension = inputs['files'].extensions[0] %}
{% else %}
{% set extension = '' %}
{% endif %}

Example::

    train_files = ["file1{{ extension }}", "file2{{ extension }}", "file3{{ extension }}", ...]
    train_targets = [0, 1, 0, ...]

    datamodule = {{ data_module_raw }}.from_files(
        train_files = train_files,
        train_targets = train_targets,
        ...
    )
{% endblock %}
{% endcall %}
{% endif %}
{% if 'datasets' in inputs %}
{% call render_subsection('from_datasets') %}

{% block from_datasets %}
Construct the {{ data_module }} from the given datasets for each stage.

Example::

    from torch.utils.data.dataset import Dataset

    train_dataset: Dataset = ...

    datamodule = {{ data_module_raw }}.from_datasets(
        train_dataset = train_dataset,
        ...
    )
{% endblock %}
{% endcall %}
{% endif %}
