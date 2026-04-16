"""
Root conftest: makes helpers / escape_helpers importable from tests/ and
patches the two gaps in decide_ai_service_base before any src/ module loads.
"""

import os
import sys

# ---------------------------------------------------------------------------
# 1. Put the project root on sys.path so `src` is importable as a package,
#    and put tests/ on sys.path so helpers.py / escape_helpers.py in that
#    directory are found before any container-injected copies.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

for _p in (_HERE, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# 2. Environment variables consumed by sparql_config and config loaders
# ---------------------------------------------------------------------------
os.environ.setdefault("MU_SPARQL_ENDPOINT", "http://localhost:8890/sparql")
os.environ.setdefault("MU_SPARQL_UPDATEPOINT", "http://localhost:8890/sparql")
os.environ.setdefault("MU_APPLICATION_GRAPH", "http://mu.semte.ch/application")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("LOG_SPARQL_ALL", "False")
os.environ.setdefault("MODE", "test")

# ---------------------------------------------------------------------------
# 3. Extend decide_ai_service_base before any src/ import
#
#    • TASK_OPERATIONS["impact_accessment"] – looked up at class-body parse
#      time in impact.py (the typo is intentional, matching the source file).
#    • SPARQL_PREFIXES["ext"] – omitted from the shipped config; store() needs
#      it to emit `PREFIX ext: <…>` via get_prefixes_for_query().
# ---------------------------------------------------------------------------
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, SPARQL_PREFIXES  # noqa: E402

TASK_OPERATIONS.setdefault(
    "impact_assessment",
    "http://lblod.data.gift/id/jobs/concept/TaskOperation/impact-assessing",
)
SPARQL_PREFIXES.setdefault("ext", "http://mu.semte.ch/vocabularies/ext/")
