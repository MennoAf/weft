"""Entry point for running the CLI: python -m weft

Prefer ``python -m weft`` (or ``uv run python -m weft``) over the
``weft`` console_script when using editable installs.  Hatchling
editable installs rely on a ``.pth`` file to add the source tree to
``sys.path``; some Python 3.13+ / uv combinations fail to process
that file when invoking the console_script directly, leading to
``ModuleNotFoundError``.  ``python -m`` always resolves the package
from the working ``sys.path`` and is therefore the reliable default.
"""

from weft.cli import cli

cli()
