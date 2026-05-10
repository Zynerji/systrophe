Systrophe
=========

**Co-rotating Tipler-cylinder pair as a tunable time-travel harness.**

*Systrophē* (Greek **Συστροφή**, "twisting-together") is the joint
exterior of two co-rotating, dual-positive-mass van Stockum dust
cylinders, whose log-periodic Tipler sinusoids superpose with a
tunable relative phase offset.

The package implements:

- The exact van Stockum interior + analytic Bonnor Case III closed forms
- A regime-dispatching robust solver for all three Bonnor regimes
- Co-axial pair (``SystrophePair``) and parallel-axis pair (``OffAxisPair``)
- N-cylinder phased array (``SystropheArray``)
- Geodesic + time-machine harness with circular orbit tuning
- The CTC zoo (Gödel, Gott, Kerr) and three singularity reinterpretations
- Energy-condition diagnostics
- Photon orbits and quantum-diagnostic primitives
- A bridge to the Δῖνος Dirac-Kerr-Newman framework

See the `whitepaper PDF <https://github.com/Zynerji/systrophe/blob/main/paper/systrophe_time_travel.pdf>`_
for a full mathematical exposition, and the
`tutorial notebook <https://github.com/Zynerji/systrophe/blob/main/examples/tutorial.ipynb>`_
for runnable examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Installation
------------

.. code-block:: bash

   pip install -e ".[dev]"
   pytest

Quick example
-------------

.. code-block:: python

   from systrophe import VanStockumInterior, find_single_cylinder_windows, harness_time_loop

   cyl = VanStockumInterior(omega=1.0, R=1.0)
   windows = find_single_cylinder_windows(cyl, r_min=1.001, r_max=200.0)
   orbit = harness_time_loop(windows[0], target_dt_per_rev=-1.0, n_revolutions=10)
   print(orbit["total_coord_time_advance"], orbit["total_proper_time_advance"])

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
