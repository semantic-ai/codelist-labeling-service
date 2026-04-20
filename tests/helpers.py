"""
Test-environment drop-in for the mu-python-template helpers module.

The real helpers.py is injected by the container at runtime and depends on
Flask's request context.  This version talks directly to Virtuoso over HTTP
(no Flask required) and accepts the `sudo` keyword argument that
decide_ai_service_base uses throughout.
"""

import logging
import os
import uuid

import requests

MU_APPLICATION_GRAPH = os.environ.get(
    "MU_APPLICATION_GRAPH", "http://mu.semte.ch/application"
)

_ENDPOINT = os.environ.get("MU_SPARQL_ENDPOINT", "http://localhost:8890/sparql")
_UPDATE_ENDPOINT = os.environ.get(
    "MU_SPARQL_UPDATEPOINT", "http://localhost:8890/sparql"
)

logger = logging.getLogger("helpers")


def query(the_query: str, sudo: bool = False) -> dict:
    """Execute a SPARQL SELECT/ASK/CONSTRUCT query and return parsed JSON."""
    resp = requests.get(
        _ENDPOINT,
        params={"query": the_query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update(the_query: str, sudo: bool = False) -> None:
    """Execute a SPARQL 1.1 Update against Virtuoso."""
    resp = requests.post(
        _UPDATE_ENDPOINT,
        data={"update": the_query},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()


def log(msg: str, *args, **kwargs) -> None:
    logger.info(msg, *args, **kwargs)


def generate_uuid() -> str:
    return str(uuid.uuid1())
