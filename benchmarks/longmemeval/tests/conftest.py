"""Reuse the project-wide pool/redis fixtures from tests/conftest.py.

The benchmark tests live outside the default ``testpaths = ["tests"]`` tree,
so pytest does not auto-discover the testcontainers session fixtures. Loading
``tests.conftest`` as a plugin pulls in ``pool`` (and the session-scoped
container start/stop hooks) without duplicating any code.
"""

# Pytest loads this conftest as the benchmark package boundary. Import the
# root fixture module under its canonical package name; unlike pytest_plugins,
# this avoids registering the same module a second time when the root conftest
# is already discovered from repository scope.
from tests.conftest import *  # noqa: F401,F403
