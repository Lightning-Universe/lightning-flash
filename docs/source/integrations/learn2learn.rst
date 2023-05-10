.. beta:: The Learn2Learn integration is currently in Beta.

.. _learn2learn:

###########
Learn2Learn
###########

`Learn2Learn <https://github.com/learnables/learn2learn>`__ is a software library for meta-learning research by `Sébastien M. R. Arnold and al.` (Aug 2020)

.. raw:: html

  <div style="margin-top: 20px; margin-bottom: 20px">
    <img src="https://raw.githubusercontent.com/learnables/learn2learn/gh-pages/assets/img/l2l-full.png" width="200px">
    <br />
  </div>



What is Meta-Learning and why you should care?
----------------------------------------------

Humans can distinguish between new objects with little or no training data,
However, machine learning models often require thousands, millions, billions of annotated data samples
to achieve good performance while extrapolating their learned knowledge on unseen objects.

A machine learning model which could learn or learn to learn from only few new samples (K-shot learning) would have tremendous applications
once deployed in production.
In an extreme case, a model performing 1-shot or 0-shot learning could be the source of new kind of AI applications.

Meta-Learning is a sub-field of AI dedicated to the study of few-shot learning algorithms.
This is often characterized as teaching deep learning models to learn with only a few labeled data.
The goal is to repeatedly learn from K-shot examples during training that match the structure of the final K-shot used in production.
It is important to note that the K-shot example seen in production are very likely to be completely out-of-distribution with new objects.


How does Meta-Learning work?
----------------------------

In meta-learning, the model is trained over multiple meta tasks.
A meta task is the smallest unit of data and it represents the data available to the model once in its deployment environment.
By doing so, we can optimise the model and get higher results.

.. raw:: html

  <div style="margin-top: 20px; margin-bottom: 20px">
    <img src="https://pl-flash-data.s3.amazonaws.com/assets/meta_learning_example.png" width="200px">
    <br />
  </div>

For image classification, a meta task is comprised of shot + query elements for each class.
The shots samples are used to adapt the parameters and the queries ones to update the original model weights.
The classes used in the validation and testing shouldn't be present within the training dataset,
as the goal is to optimise the model performance on out-of-distribution (OOD) data with little label data.

When training the model with the meta-learning algorithm,
the model will average its gradients over meta_batch_size meta tasks before performing an optimizer step.
Traditionally, an meta epoch is composed of multiple meta batch.

Use Meta-Learning with Flash
----------------------------

With its integration within Flash, Meta Learning has never been simpler.
Flash takes care of all the hard work: the tasks sampling, meta optimizer update, distributed training, etc...

.. note::

    The users requires to provide a training dataset and testing dataset with no overlapping classes.
    Flash doesn't support this feature out-of-the box.

Once done, the users are left to play the hyper-parameters associated with the meta-learning algorithm.

Here is an example using `miniImageNet dataset <https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py#L34>`_ containing 100 classes divided into 64 training, 16 validation, and 20 test classes.

.. literalinclude:: ../../../examples/image/learn2learn_img_classification_imagenette.py
    :language: python
    :lines: 15-


You can read their paper `Learn2Learn: A Library for Meta-Learning Research <https://arxiv.org/abs/2008.12284>`_.

And don't forget to cite `Learn2Learn <https://github.com/learnables/learn2learn>`__ repository in your academic publications.
Find their Biblex on their repository.
