.. customcarditem::
   :header: Point Cloud Segmentation
   :card_description: Learn to segment objects in point clouds with Flash and build an example network with the KITTI data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/point_cloud_segmentation.svg
   :tags: Point-Cloud,Segmentation

.. _pointcloud_segmentation:

########################
Point Cloud Segmentation
########################

********
The Task
********

A Point Cloud is a set of data points in space, usually describes by ``x``, ``y`` and ``z`` coordinates.

PointCloud Segmentation is the task of performing classification at a point-level, meaning each point will associated to a given class.
The current integration builds on top `Open3D-ML <https://github.com/intel-isl/Open3D-ML>`_.

------

*******
Example
*******

Let's look at an example using a data set generated from the `KITTI Vision Benchmark  <http://www.semantic-kitti.org/dataset.html>`_.
The data are a tiny subset of the original dataset and contains sequences of point clouds.
The data contains multiple folder, one for each sequence and a meta.yaml file describing the classes and their official associated color map.
A sequence should contain one folder for scans and one folder for labels, plus a ``pose.txt`` to re-align the sequence if required.
Here's the structure:

.. code-block::

    data
    ├── meta.yaml
    ├── 00
    │   ├── scans
    |   |    ├── 00000.bin
    |   |    ├── 00001.bin
    |   |    ...
    │   ├── labels
    |   |    ├── 00000.label
    |   |    ├── 00001.label
    |   |   ...
    |   ├── pose.txt
    │   ...
    |
    └── XX
       ├── scans
       |    ├── 00000.bin
       |    ├── 00001.bin
       |    ...
       ├── labels
       |    ├── 00000.label
       |    ├── 00001.label
       |   ...
       ├── pose.txt


Learn more: http://www.semantic-kitti.org/dataset.html


Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the ``PointCloudSegmentationData``.
We select a pre-trained ``randlanet_semantic_kitti`` backbone for our ``PointCloudSegmentation`` task.
We then use the trained ``PointCloudSegmentation`` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/pointcloud_segmentation.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

.. image:: https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/docs/images/getting_started_ml_visualizer.gif
   :width: 100%

------

**********
Flash Zero
**********

The point cloud segmentation task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash pointcloud_segmentation

To view configuration options and options for running the point cloud segmentation task with your own data, use:

.. code-block:: bash

    flash pointcloud_segmentation --help
