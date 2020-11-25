"""Sorts the dependencies in pyproject.toml."""
from pathlib import Path
from typing import Union

import tomlkit


def sort_table(table: tomlkit.items.Table) -> tomlkit.items.Table:
    """Sort a tomlkit table.

    converts dicts to InlineTables

    Args:
        table: Table

    Returns:
        Sorted Table
    """
    sorted_table = tomlkit.table()
    try:  # put python dependency in first positions
        python = table["python"]  # .pop() seems broken on tomlkit tables
        table.remove("python")
        sorted_table.add("python", python)
    except KeyError:
        pass
    for key, val in sorted(table.items(), key=lambda item: item[0].casefold()):
        if isinstance(val, dict):
            inline_table = tomlkit.inline_table()
            inline_table.update(val)
            val = inline_table
        sorted_table.add(key, val)
    return sorted_table


def read_toml(filename: Union[Path, str]) -> tomlkit.toml_document.TOMLDocument:
    """Read a TOML file.

    Args:
        filename: Path to the file

    Returns:
        TOML Document
    """
    with open(filename, "r") as infile:
        return tomlkit.parse(infile.read())


def dump_toml(
    filename: Union[Path, str], toml_doc: tomlkit.toml_document.TOMLDocument
) -> None:
    """Write TOML to a file.

    Args:
        filename: File path
        toml_doc: TOML document
    """
    with open(filename, "w") as f:
        f.write(tomlkit.dumps(toml_doc))


def sort_file(filename: Union[Path, str]) -> None:
    """Sort dependency sections in a file in place.

    Args:
        filename: File path
    """
    doc = read_toml(filename)
    dependencies = doc["tool"]["poetry"]["dependencies"]
    doc["tool"]["poetry"]["dependencies"] = sort_table(dependencies)
    dev_dependencies = doc["tool"]["poetry"]["dev-dependencies"]
    doc["tool"]["poetry"]["dev-dependencies"] = sort_table(dev_dependencies)
    dump_toml(filename, doc)


def sort_pyproject() -> None:
    """Sort the pyproject.toml file."""
    sort_file(Path("pyproject.toml"))


if __name__ == "__main__":
    sort_pyproject()
