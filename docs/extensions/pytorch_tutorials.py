"""Adapted from PyTorch Tutorials: https://github.com/pytorch/tutorials.

BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class cardnode(nodes.General, nodes.TextElement):
    pass


class CustomCardItemDirective(SphinxDirective):
    option_spec = {
        "header": directives.unchanged,
        "image": directives.unchanged,
        "card_description": directives.unchanged,
        "tags": directives.unchanged,
    }

    def run(self):
        try:
            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")

            if "image" in self.options:
                image = "<img src='" + self.options["image"] + "'>"
            else:
                image = "<img src='_static/images/icon.svg'>"

            # TODO: This probably only works when the tutorial list directive is in index.html
            link = self.env.docname + ".html"

            if "card_description" in self.options:
                card_description = self.options["card_description"]
            else:
                card_description = ""

            if "tags" in self.options:
                tags = self.options["tags"]
            else:
                tags = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        card_rst = CARD_TEMPLATE.format(
            header=header, image=image, link=link, card_description=card_description, tags=tags
        )
        card_list = StringList(card_rst.split("\n"))
        node = cardnode()
        self.state.nested_parse(card_list, self.content_offset, node)

        if not hasattr(self.env, "all_cardnodes"):
            self.env.all_cardnodes = []
        self.env.all_cardnodes.append({"docname": self.env.docname, "node": node})
        return [node]


CARD_TEMPLATE = """
.. raw:: html

    <div class="col-md-12 tutorials-card-container" data-tags={tags}>
    <div class="card tutorials-card">
    <a href="{link}">
    <div class="card-body">
    <div class="card-title-container">
        <h4>{header}</h4>
    </div>
    <p class="card-summary">{card_description}</p>
    <p class="tags">{tags}</p>
    <div class="tutorials-image">{image}</div>
    </div>
    </a>
    </div>
    </div>
"""


class CustomCalloutItemDirective(Directive):
    option_spec = {
        "header": directives.unchanged,
        "description": directives.unchanged,
        "button_link": directives.unchanged,
        "button_text": directives.unchanged,
    }

    def run(self):
        try:
            if "description" in self.options:
                description = self.options["description"]
            else:
                description = ""

            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")

            if "button_link" in self.options:
                button_link = self.options["button_link"]
            else:
                button_link = ""

            if "button_text" in self.options:
                button_text = self.options["button_text"]
            else:
                button_text = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        callout_rst = CALLOUT_TEMPLATE.format(
            description=description, header=header, button_link=button_link, button_text=button_text
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


CALLOUT_TEMPLATE = """
.. raw:: html

    <div class="col-md-6">
        <div class="text-container">
            <h3>{header}</h3>
            <p class="body-paragraph">{description}</p>
            <a class="btn with-right-arrow callout-button" href="{button_link}">{button_text}</a>
        </div>
    </div>
"""
