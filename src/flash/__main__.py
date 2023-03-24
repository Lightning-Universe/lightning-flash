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
import functools
import importlib
from unittest.mock import patch

import click


@click.group(no_args_is_help=True)
def main():
    """The Lightning-Flash zero-code command line utility."""


def register_command(command):
    @main.command(
        command.__name__,
        context_settings=dict(
            help_option_names=[],
            ignore_unknown_options=True,
        ),
    )
    @click.argument("cli_args", nargs=-1, type=click.UNPROCESSED)
    @functools.wraps(command)
    def wrapper(cli_args):
        with patch("sys.argv", [command.__name__] + list(cli_args)):
            command()


tasks = [
    "flash.audio.classification",
    "flash.audio.speech_recognition",
    "flash.graph.classification",
    "flash.image.classification",
    "flash.image.detection",
    "flash.image.face_detection",
    "flash.image.instance_segmentation",
    "flash.image.keypoint_detection",
    "flash.image.segmentation",
    "flash.image.style_transfer",
    "flash.pointcloud.detection",
    "flash.pointcloud.segmentation",
    "flash.tabular.classification",
    "flash.tabular.regression",
    "flash.tabular.forecasting",
    "flash.text.classification",
    "flash.text.question_answering",
    "flash.text.seq2seq.summarization",
    "flash.text.seq2seq.translation",
    "flash.video.classification",
]

for task in tasks:
    try:
        task = importlib.import_module(f"{task}.cli")

        for command in task.__all__:
            command = task.__dict__[command]
            register_command(command)
    except ImportError:
        pass

if __name__ == "__main__":
    main()
