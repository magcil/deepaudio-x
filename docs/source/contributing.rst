Contributing
============

We welcome contributions of all kinds. If you want to improve DeepAudioX, there are many ways to help.

How You Can Contribute
----------------------

- **Add a new backbone**: Implement a new backbone by following the abstract interfaces in
  ``deepaudiox.modules.baseclasses.BaseBackbone`` and register it in
  ``deepaudiox.modules.backbones``.
- **Add a new pooling strategy**: Implement a pooling module by following
  ``deepaudiox.modules.baseclasses.BasePooling`` and register it in
  ``deepaudiox.modules.pooling``.
- **Improve the library**: Optimize performance, fix bugs, enhance documentation, or add tests.

Development Setup
-----------------

Clone the repository and install all dependencies (including dev tools) using `uv`:

.. code-block:: bash

   git clone https://github.com/magcil/deepaudio-x.git
   cd deepaudio-x
   uv sync

This installs the package in editable mode along with ``pytest`` and ``ruff``.

Running Tests
-------------

.. code-block:: bash

   uv run pytest -v

General Guidelines
------------------

1. Open an issue to discuss major changes before submitting a pull request.
2. Keep changes focused and well-tested.
3. Update docs and examples when behavior changes.
4. Follow existing code style and patterns.

If you are unsure where to start, feel free to open a discussion or issue with your idea.
