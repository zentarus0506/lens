"""Shared pytest configuration.

pytest normally reuses `%TMP%/pytest-of-<user>/pytest-N`, deleting and recreating
it each run. Under sandboxed runtimes whose `mkdir` hook rejects existing paths,
that recreation raises FileExistsError before any test starts.

Pointing basetemp at a unique directory per run sidesteps the reuse entirely and
also keeps concurrent test runs from clobbering each other.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path


def pytest_configure(config) -> None:
    if getattr(config.option, "basetemp", None):
        return  # respect an explicit --basetemp
    unique = Path(tempfile.gettempdir()) / f"pytest-lens-{uuid.uuid4().hex[:8]}"
    config.option.basetemp = str(unique)
