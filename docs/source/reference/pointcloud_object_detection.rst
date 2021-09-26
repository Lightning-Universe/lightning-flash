.. customcarditem::
   :header: Point Cloud Object Detection
   :card_description: Learn to detect objects in point clouds with Flash and build an example detector with the KITTI data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/point_cloud_object_detection.svg
   :tags: Point-Cloud,Detection

.. _pointcloud_object_detection:

############################
Point Cloud Object Detection
############################

********
The Task
********

A Point Cloud is a set of data points in space, usually describes by ``x``, ``y`` and ``z`` coordinates.

PointCloud Object Detection is the task of identifying 3D objects in point clouds and their associated classes and 3D bounding boxes.

The current integration builds on top `Open3D-ML <https://github.com/intel-isl/Open3D-ML>`_.

------

*******
Example
*******

Let's look at an example using a data set generated from the `KITTI Vision Benchmark  <http://www.semantic-kitti.org/dataset.html>`_.
The data are a tiny subset of the original dataset and contains sequences of point clouds.

The data contains:
    *  one folder for scans
    *  one folder for scan calibrations
    *  one folder for labels
    *  a meta.yaml file describing the classes and their official associated color map.

Here's the structure:

.. code-block::

    data
    ├── meta.yaml
    ├── train
    │   ├── scans
    |   |    ├── 00000.bin
    |   |    ├── 00001.bin
    |   |    ...
    │   ├── calibs
    |   |    ├── 00000.txt
    |   |    ├── 00001.txt
    |   |   ...
    │   ├── labels
    |   |    ├── 00000.txt
    |   |    ├── 00001.txt
    │   ...
    ├── val
    │   ...
    ├── predict
        ├── scans
        |   ├── 00000.bin
        |   ├── 00001.bin
        |
        ├── calibs
        |   ├── 00000.txt
        |   ├── 00001.txt
        ├── meta.yaml



Learn more: http://www.semantic-kitti.org/dataset.html


Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.detection.data.PointCloudObjectDetectorData`.
We select a pre-trained ``randlanet_semantic_kitti`` backbone for our :class:`~flash.image.detection.model.PointCloudObjectDetector` task.
We then use the trained :class:`~flash.image.detection.model.PointCloudObjectDetector` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/pointcloud_detection.py
    :language: python
    :lines: 14-

.. image:: https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/docs/images/visualizer_BoundingBoxes.png
   :width: 100%

------

**********
Flash Zero
**********

The point cloud object detector can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash pointcloud_detection

To view configuration options and options for running the point cloud object detector with your own data, use:

.. code-block:: bash

    flash pointcloud_detection --help
