#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyTorchLightning/lightning-flash/blob/master/flash_notebooks/finetuning/image_classification.ipynb)

# In this notebook, we'll go over the basics of lightning Flash by finetuning an ImageClassifier on [Hymenoptera Dataset](https://www.kaggle.com/ajayrana/hymenoptera-data) containing ants and bees images.
# 
# # Finetuning
# 
# Finetuning consists of four steps:
#  
#  - 1. Train a source neural network model on a source dataset. For computer vision, it is traditionally  the [ImageNet dataset](http://www.image-net.org/search?q=cat). As training is costly, library such as [Torchvion](https://pytorch.org/docs/stable/torchvision/index.html) library supports popular pre-trainer model architectures . In this notebook, we will be using their [resnet-18](https://pytorch.org/hub/pytorch_vision_resnet/).
#  
#  - 2. Create a new neural network  called the target model. Its architecture replicates the source model and parameters, expect the latest layer which is removed. This model without its latest layer is traditionally called a backbone
#  
#  - 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.
#  
#  - 4. Train the target model on a target dataset, such as Hymenoptera Dataset with ants and bees. At training start, the backbone will be frozen, meaning its parameters won't be updated. Only the model head will be trained to properly distinguish ants and bees. On reaching first finetuning milestone, the backbone latest layers will be unfrozen and start to be trained. On reaching the second finetuning milestone, the remaining layers of the backend will be unfrozen and the entire model will be trained. In Flash, `trainer.finetune(..., unfreeze_milestones=(first_milestone, second_milestone))`.
# 
#  
# 
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [Flash documentation](https://lightning-flash.readthedocs.io/en/latest/)
#   - Check out [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)

# In[ ]:


get_ipython().run_cell_magic('capture', '', '! pip install lightning-flash')


# In[ ]:


import flash
from flash.core.data import download_data
from flash.vision import ImageClassificationData, ImageClassifier


# ## 1. Download data
# The data are downloaded from a URL, and save in a 'data' directory.

# In[ ]:


download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')


# <h2>2. Load the data</h2>
# 
# Flash Tasks have built-in DataModules that you can abuse to organize your data. Pass in a train, validation and test folders and Flash will take care of the rest.
# Creates a ImageClassificationData object from folders of images arranged in this way:</h4>
# 
# 
#    train/dog/xxx.png
#    train/dog/xxy.png
#    train/dog/xxz.png
#    train/cat/123.png
#    train/cat/nsdf3.png
#    train/cat/asd932.png
# 
# 
# Note: Each sub-folder content will be considered as a new class.

# In[ ]:


datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    valid_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
)


# ###  3. Build the model
# Create the ImageClassifier task. By default, the ImageClassifier task uses a [resnet-18](https://pytorch.org/hub/pytorch_vision_resnet/) backbone to train or finetune your model.
# For [Hymenoptera Dataset](https://www.kaggle.com/ajayrana/hymenoptera-data) containing ants and bees images, ``datamodule.num_classes`` will be 2.
# Backbone can easily be changed with `ImageClassifier(backbone="resnet50")` or you could provide your own `ImageClassifier(backbone=my_backbone)`

# In[ ]:


model = ImageClassifier(num_classes=datamodule.num_classes)


# ###  4. Create the trainer. Run once on data
# 
# The trainer object can be used for training or fine-tuning tasks on new sets of data. 
# 
# You can pass in parameters to control the training routine- limit the number of epochs, run on GPUs or TPUs, etc.
# 
# For more details, read the  [Trainer Documentation](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html).
# 
# In this demo, we will limit the fine-tuning to run just one epoch using max_epochs=2.

# In[ ]:


trainer = flash.Trainer(max_epochs=3)


# ###  5. Finetune the model
# The `unfreeze_milestones=(0, 1)` will unfreeze the latest layers of the backbone on epoch `0` and the rest of the backbone on epoch `1`. 

# In[ ]:


trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))


# ###  6. Test the model

# In[ ]:


trainer.test()


# ###  7. Save it!

# In[ ]:


trainer.save_checkpoint("image_classification_model.pt")


# <code style="color:#792ee5;">
#     <h1> <strong> Congratulations - Time to Join the Community! </strong>  </h1>
# </code>
# 
# Congratulations on completing this notebook tutorial! If you enjoyed it and would like to join the Lightning movement, you can do so in the following ways!
# 
# ### Help us build Flash by adding support for new data-types and new tasks.
# Flash aims at becoming the first task hub, so anyone can get started to great amazing application using deep learning. 
# If you are interested, please open a PR with your contributions !!! 
# 
# 
# ### Star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) on GitHub
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool tools we're building.
# 
# * Please, star [Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
# 
# ### Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!
# The best way to keep up to date on the latest advancements is to join our community! Make sure to introduce yourself and share your interests in `#general` channel
# 
# ### Interested by SOTA AI models ! Check out [Bolt](https://github.com/PyTorchLightning/pytorch-lightning-bolts)
# Bolts has a collection of state-of-the-art models, all implemented in [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and can be easily integrated within your own projects.
# 
# * Please, star [Bolt](https://github.com/PyTorchLightning/pytorch-lightning-bolts)
# 
# ### Contributions !
# The best way to contribute to our community is to become a code contributor! At any time you can go to [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) or [Bolt](https://github.com/PyTorchLightning/pytorch-lightning-bolts) GitHub Issues page and filter for "good first issue". 
# 
# * [Lightning good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * [Bolt good first issue](https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# * You can also contribute your own notebooks with useful examples !
# 
# ### Great thanks from the entire Pytorch Lightning Team for your interest !
# 
# <img src="https://github.com/PyTorchLightning/lightning-flash/blob/master/docs/source/_images/flash_logo.png?raw=true" width="800" height="200" />
