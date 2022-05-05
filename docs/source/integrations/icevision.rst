.. _ice_vision:

#########
IceVision
#########

IceVision from airctic is an awesome computer vision framework which offers a curated collection of hundreds of high-quality pre-trained models for: object detection, keypoint detection, and instance segmentation.
In Flash, we've integrated the IceVision framework to provide: data loading, augmentation, backbones, and heads.
We use IceVision components in our: :ref:`object detection <object_detection>`, :ref:`instance segmentation <instance_segmentation>`, and :ref:`keypoint detection <keypoint_detection>` tasks.
Take a look at `their documentation <https://airctic.com/>`_ and star `IceVision on GitHub <https://github.com/airctic/IceVision>`_ to spread the open source love!

IceData
_______

The `IceData library <https://github.com/airctic/icedata>`_ is a community driven dataset hub for IceVision.
All of the datasets in IceData can be used out of the box with flash using our ``.from_folders`` methods and the ``parser`` argument.
Take a look at our :ref:`keypoint_detection` page for an example.

Albumentations with IceVision and Flash
_______________________________________

IceVision provides two utilities for using the `albumentations library <https://albumentations.ai/>`_ with their models:
- the ``Adapter`` helper class for adapting an any albumentations transform to work with IceVision records,
- the ``aug_tfms`` utility function that returns a standard augmentation recipe to get the most out of your model.

In Flash, we use the ``aug_tfms`` as default transforms for the: :ref:`object detection <object_detection>`, :ref:`instance segmentation <instance_segmentation>`, and :ref:`keypoint detection <keypoint_detection>` tasks.
You can also provide custom transforms from albumentations using the :class:`~flash.core.integrations.icevision.transforms.IceVisionTransformAdapter` (which relies on the IceVision ``Adapter`` underneath).
Here's an example:

.. code-block:: python

    import albumentations as A

    from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter
    from flash.image import ObjectDetectionData

    transform = {
        "per_sample_transform": IceVisionTransformAdapter([A.HorizontalFlip(), A.Normalize()]),
    }

    datamodule = ObjectDetectionData.from_coco(
        ...,
        transform=transform,
    )
