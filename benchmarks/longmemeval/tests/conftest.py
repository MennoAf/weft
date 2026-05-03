"""Reuse the project-wide pool/redis fixtures from tests/conftest.py.

The benchmark tests live outside the default ``testpaths = ["tests"]`` tree,
so pytest does not auto-discover the testcontainers session fixtures. Loading
``tests.conftest`` as a plugin pulls in ``pool`` (and the session-scoped
container start/stop hooks) without duplicating any code.
"""

pytest_plugins = ["tests.conftest"]
