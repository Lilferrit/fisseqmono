"""Test-process setup shared by the unit and integration suites."""

import os
import sys

# On macOS, torch and xgboost each load their own OpenMP runtime; once both are imported
# into one pytest process (test_embed.py, then test_ovwt.py), xgboost segfaults on its
# first multithreaded call. One OpenMP thread avoids the clash. This has to be set before
# either library is imported, which conftest.py at the tests root guarantees. Linux (CI,
# the container) is unaffected.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
