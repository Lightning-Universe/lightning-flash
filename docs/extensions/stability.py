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
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

ADMONITION_TEMPLATE = """
.. raw:: html

    <div class="admonition warning {type}">
    <p class="admonition-title">{title}</p>
    <p>This {scope} is currently in Beta. The interfaces and functionality may change without warning in future
    releases.</p>
    </div>
"""


class Beta(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0

    def run(self):

        scope = self.arguments[0]

        admonition_rst = ADMONITION_TEMPLATE.format(type="beta", title="Beta", scope=scope)
        admonition_list = StringList(admonition_rst.split("\n"))
        admonition = nodes.paragraph()
        self.state.nested_parse(admonition_list, self.content_offset, admonition)
        return [admonition]


def setup(app):
    app.add_directive("beta", Beta)
