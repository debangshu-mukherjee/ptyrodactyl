Installation
============

Ptyrodactyl supports Python 3.12 through 3.14. Use ``pip`` to install a
published release:

.. code-block:: bash

   python -m pip install ptyrodactyl

Use ``uv`` to create the locked development environment:

.. code-block:: bash

   git clone https://github.com/debangshu-mukherjee/ptyrodactyl.git
   cd ptyrodactyl
   uv sync --extra dev

CPU execution is the reference development path. Use the ``dev_cuda`` extra
only on a supported Linux CUDA system.
