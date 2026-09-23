"""Load Chalk API credentials for the harness.

Credentials come from the environment when set, otherwise from the `chalk` CLI's
`~/.chalk.yml`: the first token entry whose apiServer matches, which is what
`chalk login` writes. Nothing here prints a secret.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml

DEFAULT_API_SERVER = "https://api.staging.chalk.dev"


def load_credentials(api_server: str, environment_id: str) -> dict[str, str]:
    """Return the env var mapping chalkcompute expects for `environment_id`."""
    client_id = os.environ.get("CHALK_CLIENT_ID")
    client_secret = os.environ.get("CHALK_CLIENT_SECRET")
    if not (client_id and client_secret):
        client_id, client_secret = _from_cli_config(api_server)
    return {
        "CHALK_API_SERVER": api_server,
        "CHALK_CLIENT_ID": client_id,
        "CHALK_CLIENT_SECRET": client_secret,
        "CHALK_ENVIRONMENT_ID": environment_id,
    }


def _from_cli_config(api_server: str) -> tuple[str, str]:
    config_path = pathlib.Path.home() / ".chalk.yml"
    if not config_path.exists():
        raise SystemExit(
            f"no CHALK_CLIENT_ID/CHALK_CLIENT_SECRET in the environment and no {config_path}; "
            "run `chalk login` or export the credentials"
        )
    config: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    for entry in (config.get("tokens") or {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("apiServer") != api_server:
            continue
        client_id = entry.get("clientId")
        client_secret = entry.get("clientSecret")
        if client_id and client_secret:
            return str(client_id), str(client_secret)
    raise SystemExit(f"no token for apiServer {api_server} in {config_path}")


def apply_to_environ(api_server: str, environment_id: str) -> None:
    """Export the credentials so the chalkcompute SDK picks them up."""
    os.environ.update(load_credentials(api_server, environment_id))
