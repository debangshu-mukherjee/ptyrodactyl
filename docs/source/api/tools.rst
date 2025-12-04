ptyrodactyl.tools
=================

.. automodule:: ptyrodactyl.tools
   :no-members:

Data Types
----------

.. autoclass:: ptyrodactyl.tools.CalibratedArray
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.CrystalStructure
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.PotentialSlices
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.ProbeModes
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.STEM4D
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.XYZData
   :members:
   :show-inheritance:

Type Aliases
------------

.. autodata:: ptyrodactyl.tools.NonJaxNumber
.. autodata:: ptyrodactyl.tools.ScalarFloat
.. autodata:: ptyrodactyl.tools.ScalarInt
.. autodata:: ptyrodactyl.tools.ScalarNumeric

Factory Functions
-----------------

.. autofunction:: ptyrodactyl.tools.make_calibrated_array
.. autofunction:: ptyrodactyl.tools.make_crystal_structure
.. autofunction:: ptyrodactyl.tools.make_potential_slices
.. autofunction:: ptyrodactyl.tools.make_probe_modes
.. autofunction:: ptyrodactyl.tools.make_stem4d
.. autofunction:: ptyrodactyl.tools.make_xyz_data

Optimizers
----------

.. autoclass:: ptyrodactyl.tools.Optimizer
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.OptimizerState
   :members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.LRSchedulerState
   :members:
   :show-inheritance:

.. autofunction:: ptyrodactyl.tools.complex_adam
.. autofunction:: ptyrodactyl.tools.complex_adagrad
.. autofunction:: ptyrodactyl.tools.complex_rmsprop
.. autofunction:: ptyrodactyl.tools.init_adam
.. autofunction:: ptyrodactyl.tools.init_adagrad
.. autofunction:: ptyrodactyl.tools.init_rmsprop
.. autofunction:: ptyrodactyl.tools.adam_update
.. autofunction:: ptyrodactyl.tools.adagrad_update
.. autofunction:: ptyrodactyl.tools.rmsprop_update
.. autofunction:: ptyrodactyl.tools.wirtinger_grad

Learning Rate Schedulers
------------------------

.. autofunction:: ptyrodactyl.tools.init_scheduler_state
.. autofunction:: ptyrodactyl.tools.create_cosine_scheduler
.. autofunction:: ptyrodactyl.tools.create_step_scheduler
.. autofunction:: ptyrodactyl.tools.create_warmup_cosine_scheduler

Loss Functions
--------------

.. autofunction:: ptyrodactyl.tools.create_loss_function

Parallel Processing
-------------------

.. autofunction:: ptyrodactyl.tools.shard_array
