.. _graph_classification:

####################
Graph Classification
####################

********
The Task
********
This task consist on classifying graphs.
The task predicts which ‘class’ the graph belongs to.
A class is a label that indicates the kind of graph.
For example, a label may indicate whether one molecule interacts with another.

The :class:`~flash.graph.classification.model.GraphClassifier` and :class:`~flash.graph.classification.data.GraphClassificationData` classes internally rely on `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`_.

------

*******
Example
*******

Let's look at the task of classifying graphs from the KKI data set from `TU Dortmund University <https://chrsmrrs.github.io/datasets>`_.

Once we've created the `TUDataset <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset>`_, we create the :class:`~flash.graph.classification.data.GraphClassificationData`.
We then create our :class:`~flash.graph.classification.model.GraphClassifier` and train on the KKI data.
Next, we use the trained :class:`~flash.graph.classification.model.GraphClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/graph_classification.py
    :language: python
    :lines: 14-
