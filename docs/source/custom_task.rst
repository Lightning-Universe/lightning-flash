
Custom tasks
############

You can create your own Flash task to if you want to solve any other deep learning problem.

See this example for defining a linear classifier task:

.. code-block:: python

	import torch
	import torch.nn.functional as F
	from flash.core.classification import ClassificationTask

	class LinearClassifier(ClassificationTask):
		def __init__(
	        self,
			num_inputs,
	        num_classes,
	        loss_fn: Callable = F.cross_entropy,
	        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
	        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
	        learning_rate: float = 1e-3,
	    ):
	        super().__init__(
	            model=None,
	            loss_fn=loss_fn,
	            optimizer=optimizer,
	            metrics=metrics,
	            learning_rate=learning_rate,
	        )
	        self.save_hyperparameters()
			
			self.linear = torch.nn.Linear(num_inputs, num_classes)
			
		def forward(self, x):
		    return self.linear(x)
