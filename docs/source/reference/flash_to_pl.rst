#######################
From Flash to Lightning
#######################

Flash is built on top of `PyTorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_ to abstract away the unnecessary boilerplate for:

- Data science
- Kaggle
- Business use cases
- Applied research

Flash is a HIGH level library and Lightning is a LOW level library.

- Flash (high-level)
- Lightning (medium-level)
- PyTorch (low-level)

As the complexity increases or decreases, users can move between Flash and Lightning seamlessly to find the
level of abstraction that works for them.

.. list-table:: Abstraction levels
   :widths: 20 20 20 20 40
   :header-rows: 1

   * - Approach
     - Flexibility
     - Minimum DL Expertise level
     - PyTorch Knowledge
     - Use cases
   * - Using an out-of-the-box task
     - Low
     - Novice+
     - Low+
     - Fast baseline, Data Science, Analysis, Applied Research
   * - Using the Generic Task
     - Medium
     - Intermediate+
     - Intermediate+
     - Fast baseline, data science
   * - Building a custom task
     - High
     - Intermediate+
     - Intermediate+
     - Fast baseline, custom business context, applied research
   * - Building a LightningModule
     - Ultimate (organized PyTorch)
     - Expert+
     - Expert+
     - For anything you can do with PyTorch, AI research (academic and corporate)

------

****************************
Using an out-of-the-box task
****************************
Tasks can come from a variety of places:

- Flash
- Other Lightning-based libraries
- Your own library

Using a task requires almost zero knowledge of deep learning and PyTorch. The focus is on solving a problem as quickly as possible.
This is great for:

- data science
- analysis
- applied research

------

**********************
Using the Generic Task
**********************
If you encounter a problem that does not have a matching task, you can use the generic task. However, this does
require a bit of PyTorch knowledge but not a lot of knowledge over all the details of deep learning.

This is great for:

- data science
- kaggle baselines
- a quick baseline
- applied research
- learning about deep learning

.. note:: If you've used something like Keras, this is the most similar level of abstraction.

------

**********************
Building a custom task
**********************
If you're feeling adventurous and there isn't an out-of-the-box task for a particular applied problem, consider
building your own task. This requires a decent amount of PyTorch knowledge, but not too much because tasks are
LightningModules that already abstract a lot of the details for you.

This is great for:

- data science
- researchers building for corporate data science teams
- applied research
- custom business context

.. note:: In a company setting, a good setup here is to have your own Flash-like library with tasks contextualized with your business problems.

------

**************************
Building a LightningModule
**************************
Once you've reached the threshold of flexibility offered by Flash, it's time to move to a LightningModule directly.
LightningModule is organized PyTorch but gives you the same flexibility. However, you must already know PyTorch
fairly well and be comfortable with at least basic deep learning concepts.

This is great for:

- experts
- academic AI research
- corporate AI research
- advanced applied research
- publishing papers
