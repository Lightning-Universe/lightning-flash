# Lightning Flash Task Template

This template is designed to guide you through implementing your own task in flash.
You should copy the files here and adapt them to your custom task.

## Required Files:

### `data.py`

Inside `data.py` you should implement:

1. one or more `DataSource` classes
2. a `Preprocess`
3. a `DataModule`
4. a `BaseVisualization` __(optional)__
5. a `Postprocess` __(optional)__

#### `DataSource`

The `DataSource` implementations describe how data from particular sources (like folders, files, tensors, etc.) should be loaded.
At a minimum you will require one `DataSource` implementation, but if you want to support a few different ways of loading data for your task, the more the merrier!
Take a look at our `TemplateDataSource` to get started.

#### `Preprocess`

The `Preprocess` is how all transforms are defined in Flash.
Internally we inject the `Preprocess` transforms into the right places so that we can address the batch at several points along the pipeline.
Defining the standard transforms (typically at least a `to_tensor_transform` and a `collate` should be defined) for your `Preprocess` is as simple as implementing the `default_transforms` method.
The `Preprocess` also knows about the available `DataSource` classes that it can work with, which should be configured in the `__init__`.
Take a look at our `TemplatePreprocess` to get started.

#### `DataModule`

The `DataModule` is where the hard work of our `DataSource` and `Preprocess` implementations pays off.
If your `DataSource` implementation(s) conform to our `DefaultDataSources` (e.g. `DefaultDataSources.FOLDERS`) then your `DataModule` implementation simply needs a `preprocess_cls` attribute.
You now have a `DataModule` that can be instantiated with `from_*` for whichever data sources you have configured (e.g. `MyDataModule.from_folders`).
It also includes all of your default transforms!

If you've defined a fully custom `DataSource`, then you will need a `preprocess_cls` attribute and one or more `from_*` methods.
The `from_*` methods take whatever arguments you want them too and call `super().from_data_source` with the name given to your custom data source in the `Preprocess.__init__`.
Take a look at our `TemplateDataModule` to get started.

#### `BaseVisualization`

A completely optional step is to implement a `BaseVisualization`. The `BaseVisualization` lets you control how data at various points in the pipeline can be visualized.
This is extremely useful for debugging purposes, allowing users to view their data and understand the impact of their transforms.
Take a look at our `TemplateVisualization` to get started, but don't worry about implementing it right away, you can always come back and add it later!

#### `Postprocess`

Sometimes you have some transforms that need to be applied _after_ your model.
For this you can optionally implement a `Postprocess`.
The `Postprocess` is applied to the model outputs during inference.
You may want to use it for: converting tokens back into text, applying an inverse normalization to an output image, resizing a generated image back to the size of the input, etc.
For information and some examples, take a look at our postprocess docs.

### `model.py`

Inside `model.py` you just need to implement your `Task`.

#### `Task`

The `Task` is responsible for the forward pass of the model.
It's just a `LightningModule` with some helpful defaults, so anything you can do inside a `LightningModule` you can do inside a `Task`.
You should configure a default loss function and optimizer and some default metrics and models in your `Task`.
Take a look at our `TemplateTask` to get started.

### `flash_examples`

Now you've implemented your task, it's time to add some examples showing how cool it is!
We usually provide one finetuning example (in `flash_examples/finetuning`) and one predict / inference example (in `flash_examples/predict`).
You can base these off of our `template.py` examples.

## Optional Files:

### `transforms.py`

Sometimes you'd like to have quite a few transforms by default (standard augmentations, normalization, etc.).
If you do then, for better organization, you can define a `transforms.py` which houses your default transforms to be referenced in your `Preprocess`.
Take a look at `vision/classification/transforms.py` for an example.

### `backbones.py`

In Flash, we love to provide as much access to the state-of-the-art as we can.
To this end, we've created the `FlashRegistry` and the backbones API.
These allow you to register backbones for your task that can be selected by the user.
The backbones can come from anywhere as long as you can register a function that loads the backbone.
If you want to configure some backbones for your task, it's best practice to include these in a `backbones.py` file.
Take a look at `vision/backbones.py` for an example, and have a look at `vision/classification/model.py` to see how these can be added to your `Task`.

### `serialization.py`

Sometimes you want to give the user some control over their prediction format.
`Postprocess` can do the heavy lifting (anything you always want to apply to the predictions), but one or more custom `Serializer` implementations can be used to convert the predictions to a desired output format.
A good example is in classification; sometimes we'd like the classes, sometimes the logits, sometimes the labels, you get the idea.
You should add your `Serializer` implementations in a `serialization.py` file and set a good default in your `Task`.
