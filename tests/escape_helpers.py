"""
Test-environment drop-in for the mu-python-template escape_helpers module.

This is a verbatim copy of the escape_helpers.py provided by the template
container, included here so the test suite can import it without needing the
container to be running.
"""

import datetime
import re
from warnings import warn


def sparql_escape_string(obj):
    """Converts the given string to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, str):
        warn(
            "You are escaping something that isn't a string with "
            "the 'sparql_escape_string'-method. Implicit casting will occur."
        )
        obj = str(obj)
    return '"""' + re.sub(r'[\\\'"]', lambda s: "\\" + s.group(0), obj) + '"""'


def sparql_escape_datetime(obj):
    """Converts the given datetime to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, datetime.datetime):
        warn(
            "You are escaping something that isn't a datetime with "
            "the 'sparql_escape_datetime'-method. Implicit casting will occur."
        )
        obj = datetime.datetime.fromisoformat(str(obj))
    return '"{}"^^xsd:dateTime'.format(obj.isoformat())


def sparql_escape_date(obj):
    """Converts the given date to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, datetime.date):
        warn(
            "You are escaping something that isn't a date with "
            "the 'sparql_escape_date'-method. Implicit casting will occur."
        )
        obj = datetime.date.fromisoformat(str(obj))
    return '"{}"^^xsd:date'.format(obj.isoformat())


def sparql_escape_time(obj):
    """Converts the given time to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, datetime.time):
        warn(
            "You are escaping something that isn't a time with "
            "the 'sparql_escape_time'-method. Implicit casting will occur."
        )
        obj = datetime.time.fromisoformat(str(obj))
    return '"{}"^^xsd:time'.format(obj.isoformat())


def sparql_escape_int(obj):
    """Converts the given int to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, int):
        warn(
            "You are escaping something that isn't an int with "
            "the 'sparql_escape_int'-method. Implicit casting will occur."
        )
        obj = int(obj)
    return '"{}"^^xsd:integer'.format(obj)


def sparql_escape_float(obj):
    """Converts the given float to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, float):
        warn(
            "You are escaping something that isn't a float with "
            "the 'sparql_escape_float'-method. Implicit casting will occur."
        )
        obj = float(obj)
    return '"{}"^^xsd:float'.format(obj)


def sparql_escape_bool(obj):
    """Converts the given bool to a SPARQL-safe RDF object string with the right RDF-datatype."""
    if not isinstance(obj, bool):
        warn(
            "You are escaping something that isn't a bool with "
            "the 'sparql_escape_bool'-method. Implicit casting will occur."
        )
        obj = bool(obj)
    return '"{}"^^xsd:boolean'.format("true" if obj else "false")


def sparql_escape_uri(obj):
    """Converts the given URI to a SPARQL-safe RDF object string."""
    obj = str(obj)
    return "<" + re.sub(r'[\\"<>]', lambda s: "\\" + s.group(0), obj) + ">"


def sparql_escape(obj):
    """Auto-selects the right escape function for the given Python value."""
    if isinstance(obj, str):
        return sparql_escape_string(obj)
    if isinstance(obj, datetime.datetime):
        return sparql_escape_datetime(obj)
    if isinstance(obj, datetime.date):
        return sparql_escape_date(obj)
    if isinstance(obj, datetime.time):
        return sparql_escape_time(obj)
    if isinstance(obj, int):
        return sparql_escape_int(obj)
    if isinstance(obj, float):
        return sparql_escape_float(obj)
    if isinstance(obj, bool):
        return sparql_escape_bool(obj)
    warn("Unknown escape type '{}'. Escaping as string".format(type(obj)))
    return sparql_escape_string(str(obj))
