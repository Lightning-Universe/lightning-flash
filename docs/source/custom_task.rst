############
Write a Task
############

If the task you wish to solve is not yes implemented, you can create your own Flash :class:`flash.core.model.Task`.

See this example for defining a linear classifier task:

.. testcode::

    import torch
    import torch.nn.functional as F
    from pytorch_lightning.metrics import Accuracy
    from flash.core.classification import ClassificationTask

    class LinearClassifier(ClassificationTask):
        def __init__(
            self,
            num_inputs: int,
            num_classes: int,
            loss_fn=F.cross_entropy,
            optimizer=torch.optim.SGD,
            metrics=[Accuracy()],
            learning_rate=1e-3,
        ):
            super().__init__(model=None,
                loss_fn=loss_fn,
                optimizer=optimizer,
                metrics=metrics,
                learning_rate=learning_rate,
            )
            self.save_hyperparameters()

            self.linear = torch.nn.Linear(num_inputs, num_classes)

        def forward(self, x):
            return self.linear(x)
