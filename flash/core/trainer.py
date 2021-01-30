import warnings
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# NOTE: copied from:
# https://github.com/PyTorchLightning/pytorch-lightning/blob/9d165f6f5655a44f1e5cd02ab36f21bc14e2a604/pl_examples/domain_templates/computer_vision_fine_tuning.py#L66
class MilestonesFinetuningCallback(BaseFinetuning):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = True):
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        # TODO: might need some config to say which attribute is model
        # maybe something like:
        # self.freeze(module=pl_module.getattr(self.feature_attr), train_bn=self.train_bn)
        # where self.feature_attr can be "backbone" or "feature_extractor", etc.
        # (configured in init)
        assert hasattr(
            pl_module, "backbone"
        ), "To use MilestonesFinetuningCallback your model must have a backbone attribute"
        self.freeze(module=pl_module.backbone, train_bn=self.train_bn)

    def finetunning_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        backbone_modules = list(pl_module.backbone.modules())
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            # TODO last N layers should be parameter
            self.unfreeze_and_add_param_group(
                module=backbone_modules[-5:],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            # TODO last N layers should be parameter
            self.unfreeze_and_add_param_group(
                module=backbone_modules[:-5],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


class Trainer(pl.Trainer):

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
    ):
        r"""
        Runs the full optimization routine. Same as pytorch_lightning.Trainer().fit()

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to fit.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped
        """
        if any(isinstance(c, BaseFinetuning) for c in self.callbacks):
            # TODO: if we find a finetuning callback in the trainer should we remove it? or just warn the user?
            warnings.warn("Warning: You are calling fit(), but your trainer is using a fine-tuning callback")
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    def finetune(
        self,
        model: pl.LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        unfreeze_milestones: tuple = (5, 10),
    ):
        r"""
        Runs the full optimization routine. Same as pytorch_lightning.Trainer().fit(), but unfreezes layers
        of the backbone throughout training layers of the backbone throughout training.

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to fit.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            unfreeze_milestones: A tuple of two integers. First value marks the epoch in which the last 5
            layers of the backbone will be unfrozen. The second value marks the epoch in which the full backbone will
            be unfrozen.

        """
        if hasattr(model, "backbone"):
            # TODO: if we find a finetuning callback in the trainer should we change it?
            # or should we warn the user?
            if not any(isinstance(c, BaseFinetuning) for c in self.callbacks):
                # TODO: should pass config from arguments
                self.callbacks.append(MilestonesFinetuningCallback(milestones=unfreeze_milestones))
        else:
            warnings.warn("Warning: model does not have a 'backbone' attribute, will train normally")

        return super().fit(model, train_dataloader, val_dataloaders, datamodule)
