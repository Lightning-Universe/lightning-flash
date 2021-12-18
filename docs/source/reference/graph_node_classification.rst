.. customcarditem::
   :header: Graph Node Classification
   :card_description: Learn to classify nodes in graphs with Flash and build an example classifier for the Planetoid Cora data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/graph_classification.svg
   :tags: Graph,Classification

.. _graph_node_classification:

####################
Graph Node Classification
####################

********
The Task
********
This task consist on classifying nodes from graphs.
The task predicts which ‘class’ each node belongs to.
A class is a label that indicates the kind of node.
For example, a label may indicate whether one author is belongs to a field of study, based on who he collaborated with.

The :class:`~flash.graph.classification.model.GraphNodeClassifier` and :class:`~flash.graph.classification.data.GraphClassificationData` classes internally rely on `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`_.

------

*******
Example
*******

Let's look at the task of classifying graphs from the Planetoid data set from `TU Dortmund University <https://chrsmrrs.github.io/datasets>`_.
Once we've created the `Planetoid <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid>`_, we create the :class:`~flash.graph.classification.data.GraphClassificationData`.
We then create our :class:`~flash.graph.classification.model.GraphNodeClassifier` and train on the Planetoid (Cora) data.
Next, we use the trained :class:`~flash.graph.classification.model.GraphNodeClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/graph_node_classification.py
    :language: python
    :lines: 14-

------

**********
Flash Zero
**********

The graph classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash graph_node_classification

To view configuration options and options for running the graph classifier with your own data, use:

.. code-block:: bash

    flash graph_node_classification --help
