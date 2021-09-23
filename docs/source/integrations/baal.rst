.. _baal:

####
BaaL
####

The framework `Bayesian Active Learning (BaaL) <https://github.com/ElementAI/baal>`_  is an active learning
library developed at `ElementAI <https://www.elementai.com/>`_.

.. raw:: html

  <div style="margin-top: 20px; margin-bottom: 20px">
    <img src="https://raw.githubusercontent.com/ElementAI/baal/master/docs/_static/images/logo-transparent.png" width="100px">
  </div>


Active Learning is a sub-field in AI, focusing on adding a human in the learning loop.
The most uncertain samples will be labelled by the human to accelerate the model training cycle.

.. raw:: html

  <div style="margin-top: 20px; margin-bottom: 20px">
    <img src="https://raw.githubusercontent.com/ElementAI/baal/master/docs/literature/images/Baalscheme.svg" width="400px">
    <p align="center">Credit to ElementAI / Baal Team for creating this diagram flow</p>
    <br />
  </div>

With its integration within Flash, Active Learning process is made even simpler than before.


.. literalinclude:: ../../../flash_examples/integrations/baal/image_classification_active_learning.py
    :language: python
    :lines: 14-
