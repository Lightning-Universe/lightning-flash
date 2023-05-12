# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from flash import Trainer
from flash.core.data.utils import download_data
from flash.text import SummarizationData, SummarizationTask

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", "./data/")

datamodule = SummarizationData.from_csv(
    "input",
    "target",
    train_file="data/xsum/train.csv",
    val_file="data/xsum/valid.csv",
    batch_size=4,
)

# 2. Build the task
model = SummarizationTask()

# 3. Create the trainer and finetune the model
trainer = Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Summarize some text!
datamodule = SummarizationData.from_lists(
    predict_data=[
        """
        Camilla bought a box of mangoes with a Brixton Â£10 note, introduced last year to try to keep the money of local
        people within the community.The couple were surrounded by shoppers as they walked along Electric Avenue.
        They came to Brixton to see work which has started to revitalise the borough.
        It was Charles' first visit to the area since 1996, when he was accompanied by the former
        South African president Nelson Mandela.Greengrocer Derek Chong, who has run a stall on Electric Avenue
        for 20 years, said Camilla had been ""nice and pleasant"" when she purchased the fruit.
        ""She asked me what was nice, what would I recommend, and I said we've got some nice mangoes.
        She asked me were they ripe and I said yes - they're from the Dominican Republic.""
        Mr Chong is one of 170 local retailers who accept the Brixton Pound.
        Customers exchange traditional pound coins for Brixton Pounds and then spend them at the market
        or in participating shops.
        During the visit, Prince Charles spent time talking to youth worker Marcus West, who works with children
        nearby on an estate off Coldharbour Lane. Mr West said:
        ""He's on the level, really down-to-earth. They were very cheery. The prince is a lovely man.""
        He added: ""I told him I was working with young kids and he said, 'Keep up all the good work.'""
        Prince Charles also visited the Railway Hotel, at the invitation of his charity The Prince's Regeneration Trust.
        The trust hopes to restore and refurbish the building,
        where once Jimi Hendrix and The Clash played, as a new community and business centre."
        """,
        """
        "The problem is affecting people using the older versions of the PlayStation 3, called the ""Fat"" model. The
        problem isn't affecting the newer PS3 Slim systems that have been on sale since September last year. Sony have
        also said they are aiming to have the problem fixed shortly but is advising some users to avoid using their
        console for the time being.""We hope to resolve this problem within the next 24 hours,"" a statement reads.
        ""In the meantime, if you have a model other than the new slim PS3, we advise that you do not use your PS3
        system, as doing so may result in errors in some functionality, such as recording obtained trophies, and not
        being able to restore certain data.""We believe we have identified that this problem is being caused by a bug
        in the clock functionality incorporated in the system.""The PlayStation Network is used by millions of people
        around the world.It allows users to play their friends at games like Fifa over the internet and also do things
        like download software or visit online stores.",Sony has told owners of older models of its PlayStation 3
        console to stop using the machine because of a problem with the PlayStation Network.
        """,
    ],
    batch_size=4,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("summarization_model_xsum.pt")
