Installation
============

This section provides instructions on how to install DeepAudio-X.

Requirements
------------

DeepAudio-X requires Python 3.11, 3.12 or 3.13. It is recommended to use a virtual environment.

With venv:

.. code-block:: bash

   python3 -m venv deepaudiox-env
   source deepaudiox-env/bin/activate  # On Windows use `deepaudiox-env\Scripts\activate`

With miniconda:

.. code-block:: bash

   conda create -n deepaudiox-env python=3.13
   conda activate deepaudiox-env

With `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ (recommended):

.. code-block:: bash

   uv venv --python 3.13
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

PyPI
----

DeepAudio-X is available on PyPI. Install with pip:

.. code-block:: bash

   pip install deepaudio-x

Or with uv:

.. code-block:: bash

   uv pip install deepaudio-x

.. note::

   The PyTorch version pulled from PyPI may require a newer NVIDIA driver than
   what is installed on your system. After installation, verify that CUDA is available:

   .. code-block:: python

      import torch
      print(torch.cuda.is_available())   # True if GPU is available
      print(torch.version.cuda)          # CUDA version PyTorch was built for

   If ``False`` is returned, either update your driver from
   `nvidia.com <https://www.nvidia.com/Download/index.aspx>`_ or downgrade PyTorch
   to a version compatible with your driver. See
   `pytorch.org <https://pytorch.org/get-started/locally/>`_ to find the right build.

Source
------

If you want a pre-release version, clone the repo and use `uv sync` to install
dependencies from `pyproject.toml` and `uv.lock`. For installing uv itself, see
the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

.. code-block:: bash

   git clone https://github.com/magcil/deepaudio-x.git
   cd deepaudio-x
   uv sync

Verify the installation:

.. code-block:: bash

   python -c "import deepaudiox; print(deepaudiox.__version__)"
