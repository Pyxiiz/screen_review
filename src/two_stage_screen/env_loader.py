"""Load local secrets from a `.env` file (gitignored); see `.env.example` in the project root."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_project_dotenv() -> None:
    """
    Load the nearest `.env` walking up from the current working directory.
    Existing environment variables are not overridden (same as python-dotenv default).
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)
    else:
        load_dotenv()
