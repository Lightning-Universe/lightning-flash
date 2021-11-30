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
import importlib
from argparse import REMAINDER
from unittest import mock

import jsonargparse

from flash.core.utilities.lightning_cli import _get_short_description


def main() -> None:
    parser = jsonargparse.ArgumentParser(description="The Lightning Flash zero-code command line utility.")

    # list of supported CLI tasks, the format is `[subcommand]: class_path`
    tasks = {
        "image_classification": "flash.image.classification.cli",
        # FIXME: add support for all tasks
        # "flash.audio.classification",
        # "flash.audio.speech_recognition",
        # "flash.graph.classification",
        # "flash.image.detection",
        # "flash.image.instance_segmentation",
        # "flash.image.keypoint_detection",
        # "flash.image.segmentation",
        # "flash.image.style_transfer",
        # "flash.pointcloud.detection",
        # "flash.pointcloud.segmentation",
        # "flash.tabular.classification",
        # "flash.tabular.forecasting",
        # "flash.text.classification",
        # "flash.text.question_answering",
        # "flash.text.seq2seq.summarization",
        # "flash.text.seq2seq.translation",
        # "flash.video.classification",
    }

    # add the subcommands for each task, with their respective arguments
    parser_subcommands = parser.add_subcommands()
    for name, class_path in tasks.items():
        task_cli_module = importlib.import_module(class_path)
        cli_fn = getattr(task_cli_module, "cli")  # assumes the function is called just `cli`
        description = _get_short_description(cli_fn)
        subcommand_parser = jsonargparse.ArgumentParser()
        subcommand_parser.add_function_arguments(cli_fn)
        subcommand_parser.add_argument("task_args", nargs=REMAINDER)
        parser_subcommands.add_subcommand(name, subcommand_parser, help=description)

    args = parser.parse_args()

    # wrap and call the function
    selected = tasks[args.subcommand]
    task_args = args[args.subcommand].pop("task_args")
    selected_cli_module = importlib.import_module(selected)
    cli_fn = getattr(selected_cli_module, "cli")  # assumes the function is called just `cli`
    with mock.patch("sys.argv", ["flash.py"] + task_args):
        cli_kwargs = args[args.subcommand]
        cli_fn(**cli_kwargs)


if __name__ == "__main__":
    main()
