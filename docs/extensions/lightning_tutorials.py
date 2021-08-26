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
from docutils.statemachine import StringList
from pytorch_tutorials import cardnode, CustomCalloutItemDirective, CustomCardItemDirective
from sphinx.util.docutils import SphinxDirective


class tutoriallistnode(nodes.General, nodes.Element):
    pass


def visit_cardnode(self, node):
    self.visit_paragraph(node)


def depart_cardnode(self, node):
    self.depart_paragraph(node)


def purge_cards(app, env, docname):
    if not hasattr(env, "all_cardnodes"):
        return

    env.all_cardnodes = [node for node in env.all_cardnodes if node["docname"] != docname]


def merge_cards(app, env, docnames, other):
    if not hasattr(env, "all_cardnodes"):
        env.all_cardnodes = []
    if hasattr(other, "all_cardnodes"):
        env.all_cardnodes.extend(other.all_cardnodes)


TUTORIAL_LIST_START = """
.. raw:: html

   </div>
   <div id="tutorial-cards-container">

   <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
     <div class="tutorial-tags-container">
         <div id="dropdown-filter-tags">
             <div class="tutorial-filter-menu">
                 <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
             </div>
         </div>
     </div>
   </nav>

   <hr class="tutorials-hr">

   <div class="row">

   <div id="tutorial-cards">
   <div class="list">

.. Tutorial cards below this line

"""

TUTORIAL_LIST_END = """

.. End of tutorial card section

.. raw:: html

   </div>

   <div class="pagination d-flex justify-content-center"></div>

   </div>

   </div>
   <br>
   <br>
"""


class TutorialListDirective(SphinxDirective):
    def run(self):
        start_list = StringList(TUTORIAL_LIST_START.split("\n"))
        start_node = nodes.paragraph()
        self.state.nested_parse(start_list, self.content_offset, start_node)

        end_list = StringList(TUTORIAL_LIST_END.split("\n"))
        end_node = nodes.paragraph()
        self.state.nested_parse(end_list, self.content_offset, end_node)

        return start_node.children + [tutoriallistnode("")] + end_node.children


def process_card_nodes(app, doctree, fromdocname):
    for card in doctree.traverse(cardnode):
        card.parent.remove(card)

    for node in doctree.traverse(tutoriallistnode):
        content = []

        for card in app.builder.env.all_cardnodes:
            content.append(card["node"])

        node.replace_self(content)


def setup(app):
    app.add_node(tutoriallistnode)
    app.add_node(
        cardnode,
        html=(visit_cardnode, depart_cardnode),
        latex=(visit_cardnode, depart_cardnode),
        text=(visit_cardnode, depart_cardnode),
    )

    app.add_directive("customcarditem", CustomCardItemDirective)
    app.add_directive("customcalloutitem", CustomCalloutItemDirective)
    app.add_directive("tutoriallist", TutorialListDirective)

    app.connect("env-purge-doc", purge_cards)
    app.connect("env-merge-info", merge_cards)
    app.connect("doctree-resolved", process_card_nodes)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
