ptyrodactyl.tools
=================

Optimizers
----------

.. autoclass:: ptyrodactyl.tools.Optimizer
   :no-members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.OptimizerState
   :no-members:
   :show-inheritance:

.. autoclass:: ptyrodactyl.tools.LRSchedulerState
   :no-members:
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
