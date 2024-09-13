
.. image:: https://raw.githubusercontent.com/buettnerlab/convexgating/add_parameter_option/figures/CG_logo_v2.PNG
   :width: 800
   :alt: overview


|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/convexgating.svg
   :target: https://pypi.org/project/convexgating/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/convexgating
   :target: https://pypi.org/project/convexgating
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/buettnerlab/convexgating
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/convexgating/latest.svg?label=Read%20the%20Docs
   :target: https://convexgating.readthedocs.io/
   :alt: Read the documentation at https://convexgating.readthedocs.io/
.. |Build| image:: https://github.com/buettnerlab/convexgating/workflows/Build%20convexgating%20Package/badge.svg
   :target: https://github.com/buettnerlab/convexgating/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/buettnerlab/convexgating/workflows/Run%20convexgating%20Tests/badge.svg
   :target: https://github.com/buettnerlab/convexgating/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/buettnerlab/convexgating/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/buettnerlab/convexgating
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

Convex gating is a Python package to infer an optimal gating strategy from flow, cyTOF or Ab/CITE-seq data. Convex gating expects a labelled input (for instance, from clustering) and returns a gating panel to separate the selected group of events (e.g. a cluster) from all other events (see Fig. 1a).
For each cluster, it reports the purity (precision), yield (recall) and the harmonic mean of both metrics (F1 score) for each gate hierarchy and the entire gating strategy. It relies on the scanpy/anndata for the data format and data pre-processing and further on PyTorch for stochastic gradient descent. Therefore, resulting gates may slightly vary.

.. image:: https://raw.githubusercontent.com/buettnerlab/convexgating/main/figures/fig1_v4.PNG
   :width: 800
   :alt: overview

The iterative procedure to find a suitable gate before applying the convex hull is illustrated in the following graphic.


.. image:: https://raw.githubusercontent.com/buettnerlab/convexgating/main/figures/fig_update_step_v5.png
   :width: 800
   :alt: Update


Installation
------------
.. code:: console

   $ git clone https://github.com/buettnerlab/convexgating.git
   $ cd convexgating
   $ pip install -e .




Usage
-----

Please see the `Command-line Reference <Usage_>`_ for details.



Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://convexgating.readthedocs.io/en/latest/usage.html
