.. _graph_classification:

####################
Graph Embedding
####################

********
The Task
********
This task consists of creating an embedding of a graph. That is, a vector of features which can be used for a downstream task.
The :class:`~flash.graph.classification.model.GraphEmbedder` and :class:`~flash.graph.classification.data.GraphClassificationData` classes internally rely on `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`_.

------

*******
Example
*******

Let's look at the task of classifying graphs from the KKI data set from `TU Dortmund University <https://chrsmrrs.github.io/datasets>`_.

Once we've created the `TUDataset <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset>`, we create the :class:`~flash.graph.classification.data.GraphClassificationData`.
We then create our :class:`~flash.graph.classification.model.GraphEmbedder`.
Then, we load a trained :class:`~flash.graph.classification.model.GraphEmbedder` (for example from a previously trained :class:`~flash.graph.classification.model.GraphClassifier`).
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/graph_classification.py
    :language: python
    :lines: 14
