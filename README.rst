.. -*- mode: rst -*-

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/sklearn-upgrade.svg?
   :target: https://pypi.org/project/sklearn-upgrade/

.. |PyPI| image:: https://img.shields.io/pypi/v/sklearn-upgrade
   :target: https://pypi.org/project/sklearn-upgrade

.. |PythonMinVersion| replace:: 3.11
.. |NumPyMinVersion| replace:: 1.24.1
.. |SciPyMinVersion| replace:: 1.10.0
.. |JoblibMinVersion| replace:: 1.3.0
.. |NarwhalsMinVersion| replace:: 2.0.1
.. |ThreadpoolctlMinVersion| replace:: 3.2.0
.. |MatplotlibMinVersion| replace:: 3.6.1
.. |Scikit-ImageMinVersion| replace:: 0.22.0
.. |PandasMinVersion| replace:: 1.5.0
.. |SeabornMinVersion| replace:: 0.13.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.18.0

**sklearn-upgrade** is a modified version of scikit-learn – a Python module for machine learning built on top of SciPy and distributed under the 3-Clause BSD license.

This fork adds custom improvements and enhancements over the original scikit-learn. The original project was started in 2007 by David Cournapeau as a Google Summer of Code project and has since received contributions from many volunteers.

Installation
------------

Dependencies
~~~~~~~~~~~~

sklearn-upgrade requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- Narwhals (>= |NarwhalsMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

Scikit-learn plotting capabilities (i.e., functions starting with ``plot_`` and classes ending with ``Display``) require Matplotlib (>= |MatplotlibMinVersion|). For running the examples Matplotlib >= |MatplotlibMinVersion| is required. A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples require pandas >= |PandasMinVersion|, some examples require seaborn >= |SeabornMinVersion| and Plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy, the easiest way to install sklearn-upgrade is using ``pip``::

    pip install -U sklearn-upgrade

or ``conda`` (if available on conda-forge)::

    conda install -c conda-forge sklearn-upgrade

Changelog
---------

See the `changelog <https://github.com/ProdUssr/scikit-learn-upgrade/releases>`__ for a history of notable changes to sklearn-upgrade.

Development
-----------

We welcome new contributors of all experience levels. The sklearn-upgrade community goals are to be helpful, welcoming, and effective. The original scikit-learn `Development Guide <https://scikit-learn.org/stable/developers/index.html>`_ has detailed information about contributing code, documentation, tests, and more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/ProdUssr/scikit-learn-upgrade
- Download releases: https://pypi.org/project/sklearn-upgrade/
- Issue tracker: https://github.com/ProdUssr/scikit-learn-upgrade/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/ProdUssr/scikit-learn-upgrade.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to sklearn-upgrade, please see our `Contributing guide <https://github.com/ProdUssr/scikit-learn-upgrade/blob/main/CONTRIBUTING.md>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source directory (you will need to have ``pytest`` >= |PytestMinVersion| installed)::

    pytest sklearn_upgrade

Random number generation can be controlled during testing by setting the ``SKLEARN_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the full Contributing page to make sure your code complies with our guidelines: https://github.com/your-username/sklearn-upgrade/blob/main/CONTRIBUTING.md

Project History
---------------

This package is a fork of the original `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_. The original project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the `About us <https://scikit-learn.org/dev/about.html#authors>`__ page for a list of core contributors.

The original project is maintained by a team of volunteers. sklearn-upgrade is maintained independently and may diverge from upstream.

**Note**: `scikit-learn` was previously referred to as `scikits.learn`.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- Original scikit-learn documentation (stable release): https://scikit-learn.org
- Original scikit-learn documentation (development version): https://scikit-learn.org/dev/
- FAQ: https://scikit-learn.org/stable/faq.html

For sklearn-upgrade specific documentation, please refer to the `GitHub repository <https://github.com/ProdUssr/scikit-learn-upgrade>`_.

Communication
~~~~~~~~~~~~~

Main Channels (original project)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Website**: https://scikit-learn.org
- **Blog**: https://blog.scikit-learn.org
- **Mailing list**: https://mail.python.org/mailman/listinfo/scikit-learn

For issues or discussions about sklearn-upgrade, please use our `GitHub Discussions <https://github.com/ProdUssr/scikit-learn-upgrade/discussions>`_ or `issue tracker <https://github.com/ProdUssr/scikit-learn-upgrade/issues>`_.

Citation
~~~~~~~~

If you use sklearn-upgrade in a scientific publication, please cite the original scikit-learn paper: https://scikit-learn.org/stable/about.html#citing-scikit-learn