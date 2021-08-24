{% extends "base.rst" %}
{% block from_datasets %}
{{ super() }}

.. note::

    The ``__getitem__`` of your datasets should return a dictionary with ``"input"`` and ``"target"`` keys which map to the input spectrogram image (as a NumPy array) and the target (as an int or list of ints) respectively.
{% endblock %}
