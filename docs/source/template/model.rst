.. _contributing_task:

********
The Task
********

Inside ``model.py`` you just need to implement your ``Task``.
The `Task` is responsible for the forward pass of the model.
It's just a `LightningModule` with some helpful defaults, so anything you can do inside a `LightningModule` you can do inside a `Task`.

Task
^^^^

You should configure a default loss function and optimizer and some default metrics and models in your `Task`.
Take a look at our `TemplateTask` to get started.
