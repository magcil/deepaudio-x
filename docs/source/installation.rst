Installation
============

PyPI
----

DeepAudio-X is available on PyPI and can be installed with pip:

.. code-block:: bash

   pip install deepaudio-x

Install a pre-release from source with uv
-----------------------------------------

If you want a pre-release version, clone the repo and use `uv sync` to install
dependencies from `pyproject.toml` and `uv.lock`. For installing uv itself, see
the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

.. code-block:: bash

   git clone https://github.com/magcil/deepaudio-x.git
   cd deepaudio-x
   uv sync
