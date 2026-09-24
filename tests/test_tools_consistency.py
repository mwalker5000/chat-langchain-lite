"""Guards that the curated knowledge base in agent/tools.py agrees with itself."""

import re

from agent.tools import CONCEPTS_DB, SETUP_GUIDES_DB

_MINIMUM_SENTENCE = re.compile(r"Minimum supported Python is ([^\n]+)")
_VERSION_FOR_PACKAGES = re.compile(r"(\d+\.\d+) for ([a-z/ ]+)")


def _declared_minimums():
    sentence = _MINIMUM_SENTENCE.search(SETUP_GUIDES_DB["installation"])
    assert sentence, "installation guide no longer states a minimum Python version"
    minimums = {}
    for version, packages in _VERSION_FOR_PACKAGES.findall(sentence.group(1)):
        for package in packages.replace(" and ", "/").split("/"):
            if package.strip():
                minimums[package.strip()] = version
    return minimums


def _version_tuple(value):
    return tuple(int(part) for part in value.rstrip("+").split("."))


def test_concept_min_python_matches_installation_guide():
    declared = _declared_minimums()
    assert declared, "no per-package minimums parsed from the installation guide"
    floor = min(_version_tuple(version) for version in declared.values())

    for concept, data in CONCEPTS_DB.items():
        package = data["package"].split()[0]
        min_python = data["min_python"]
        if package in declared:
            expected = f"{declared[package]}+"
            assert min_python == expected, (
                f"CONCEPTS_DB['{concept}'] claims Python {min_python} for "
                f"{package}, but the installation guide says {expected}"
            )
        else:
            assert _version_tuple(min_python) >= floor, (
                f"CONCEPTS_DB['{concept}'] claims Python {min_python}, below the "
                f"lowest version the installation guide supports"
            )
