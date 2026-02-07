Installation
============

This section provides instructions on how to install DeepAudio-X.

Requirements
------------

DeepAudio-X requires Python 3.11, 3.12 or 3.13. It is recommended to use a virtual environment. For example, you can use miniconda or venv.

Example with venv:

.. code-block:: bash

   python3 -m venv deepaudiox-env
   source deepaudiox-env/bin/activate  # On Windows use `deepaudiox-env\Scripts\activate`

Or with miniconda:

.. code-block:: bash

   conda create -n deepaudiox-env python=3.13
   conda activate deepaudiox-env

PyPI
----

DeepAudio-X is available on PyPI and can be installed with pip:

.. code-block:: bash

   pip install deepaudio-x

Source
-----------------------------------------

If you want a pre-release version, clone the repo and use `uv sync` to install
dependencies from `pyproject.toml` and `uv.lock`. For installing uv itself, see
the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

.. code-block:: bash

   git clone https://github.com/magcil/deepaudio-x.git
   cd deepaudio-x
   uv sync
