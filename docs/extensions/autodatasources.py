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
import os

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from jinja2 import ChoiceLoader, Environment, FileSystemLoader, FunctionLoader
from sphinx.util.nodes import nested_parse_with_titles


class TemplateLoader:
    def __init__(self):
        self.templates = {}

    def __call__(self, name: str):
        return self.templates.get(name, None)

    def add_template(self, name: str, content: str):
        self.templates[name] = content


_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_LOADER = TemplateLoader()
ENVIRONMENT = Environment(
    loader=ChoiceLoader([FileSystemLoader(os.path.join(_PATH_HERE, "templates")), FunctionLoader(TEMPLATE_LOADER)])
)


class AutoInputs(Directive):
    has_content = True
    required_arguments = 2
    optional_arguments = 0

    def run(self):
        data_module_path, data_module_name = self.arguments

        string_list = self.content
        source = string_list.source(0)
        TEMPLATE_LOADER.add_template(data_module_name, "\n".join(string_list.data))

        data_module = getattr(importlib.import_module(data_module_path), data_module_name)

        class PatchedPreprocess(data_module.preprocess_cls):
            """TODO: This is a hack to prevent default transforms form being created"""

            @staticmethod
            def _resolve_transforms(_):
                return None

        preprocess = PatchedPreprocess()
        data_sources = {
            data_source: preprocess.data_source_of_name(data_source)
            for data_source in preprocess.available_data_sources()
        }

        ENVIRONMENT.get_template("base.rst")

        rendered_content = ENVIRONMENT.get_template(data_module_name).render(
            data_module=f":class:`~{data_module_path}.{data_module_name}`",
            data_module_raw=data_module_name,
            data_sources=data_sources,
        )

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, StringList(rendered_content.split("\n"), source), node)
        return node.children


def setup(app):
    app.add_directive("autoinputs", AutoInputs)
