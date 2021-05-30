.. _graph_classification:

####################
Graph Classification
####################

********
The task
********
This task consist on classifying graphs. The task predicts which ‘class’ the graph most likely belongs to with a degree of certainty.  A class is a label that indicates the kind of graph. For example, a label may indicate whether a molecule is predicted to interact with another.

------

*********
Inference
*********

We can use a trained model (for example loading it from a checkpoint) to perform inference

.. note:: We skip the first 14 lines as they are just the copyright notice.

One example from a `TUDataset <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset>`

.. literalinclude:: ../../../flash_examples/predict/graph_classification.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

********
Training
********

Before we make predictions, we also must train our model. We can once again use a `TUDataset <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset>`
Notice that the GraphDataSource will take the data from a `Pytorch Geometric Dataset <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>`, either downloaded or hand-made.

.. literalinclude:: ../../../flash_examples/finetuning/graph_classification.py
    :language: python
    :lines: 14-

------

*************
API reference
*************

.. _graph_data:

GraphClassificationData
-----------

.. autoclass:: flash.graph.classification.GraphClassificationData

.. automethod:: flash.graph.classification.GraphClassificationData.from_data_source
