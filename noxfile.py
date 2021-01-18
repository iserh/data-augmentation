"""File containing all nox session definitions."""
import configparser

import nox
from nox.sessions import Session

python_versions = ["3.8"]
nox.options.sessions = "fmt", "lint"
locations = ("vae", "utils", "noxfile.py", "mlflow_gc.py")

# load line length from flake8 config
config = configparser.ConfigParser()
config.read(".flake8")
max_line_length = config["flake8"]["max-line-length"]


@nox.session(python=[python_versions])
def lint(session: Session) -> None:
    """Run flake8 linting.

    Args:
        session (Session): Nox session
    """
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-builtins",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-isort",
        "flake8-use-fstring",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=[python_versions])
def black(session: Session) -> None:
    """Run black formatter.

    Args:
        session (Session): Nox session
    """
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python=[python_versions])
def isort(session: Session) -> None:
    """Run isort input sorting.

    Args:
        session (Session): Nox session
    """
    args = session.posargs or locations
    session.install("isort")
    session.run("isort", "--atomic", *args)


@nox.session(python=[python_versions])
def fmt(session: Session) -> None:
    """Autoformat project using black and isort.

    Args:
        session (Session): Nox session
    """
    args = session.posargs or locations
    session.install("isort", "black", "docformatter", "reindent")
    session.run("isort", "--atomic", *args)
    session.run(
        "docformatter",
        "--wrap-summaries",
        f"{max_line_length}",
        "--wrap-descriptions",
        f"{max_line_length}",
        "--in-place",
        "--recursive",
        *args,
    )
    session.run("python", "-m", "reindent", "-r", "-n", *args)
    session.run("black", *args)
