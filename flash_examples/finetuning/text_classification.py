# import our libraries
import torch

import flash
from flash.core.data import download_data
from flash.text import TextClassificationData, TextClassifier

# 1. Download data
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

# 2. Organize our data
datamodule = TextClassificationData.from_files(
    train_file="data/imdb/train.csv",
    valid_file="data/imdb/valid.csv",
    input="review",
    target="sentiment",
)

# 3. Build a model
model = TextClassifier(num_classes=datamodule.num_classes)

# 4. Create trainer - Make training slightly faster for demo.
trainer = flash.Trainer(max_epochs=1, limit_train_batches=8, limit_val_batches=8)

# 5. Finetune the model
trainer.finetune(model, datamodule=datamodule)

# 6. Save model
torch.save(model, "text_classification_model.pt")
