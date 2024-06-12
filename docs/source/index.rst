.. meent documentation master file, created by
   sphinx-quickstart on Tue Jun 11 15:23:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Meent: Electromagnetic simulation
===================================

Meent is a Python library for electromagnetic simulation using RCWA, designed for
Machine Learning integration.

.. `https://github.com/kc-ml2/meent <https://github.com/kc-ml2/meent>`_ is link.

Check out the usage section for further information, including
how to installation the project.

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
        :columns: 12 6 6 4
        :link: getting-started
        :link-type: ref
        :class-card: getting-started

    .. grid-item-card:: :material-regular:`description;2em` Theories
      :columns: 12 6 6 4
      :link: theories
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`terminal;2em` Developer Docs
      :columns: 12 6 6 4
      :link: contributor-guide
      :link-type: ref
      :class-card: developer-docs

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Getting Started

    getting-started
    tutorials/README.md
    tutorials

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Theories

   arxiv-paper
   theories

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer Docs

   sequence/seq
   meent

.. image:: /_static/meent-summary.png
    :alt: Meent summary



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
