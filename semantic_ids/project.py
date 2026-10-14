"""Small local receipts for a user's explicit commands; no remote controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / ".semantic-ids" / "project.json"


@dataclass(frozen=True)
class Target:
    environment: str
    api_server: str

    def __post_init__(self):
        if not self.environment.strip():
            raise ValueError("An explicit Chalk environment ID is required")
        url = urlsplit(self.api_server)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("The Chalk API server must be an HTTPS origin")

    @classmethod
    def selected(cls, environment: str, api_server: str | None = None):
        # Use Chalk's own repository-aware authentication. Do not scrape tokens
        # or choose the first credential for an API host.
        from chalk.config.auth_config import load_token
        if api_server is None:
            api_server = os.environ.get("CHALK_API_SERVER")
        if api_server is None:
            token = load_token(client_id=None, client_secret=None,
                               active_environment=environment, api_server=None, skip_cache=True)
            if token is None:
                raise ValueError("Configure Chalk authentication in this repository first")
            api_server = token.apiServer
        return cls(environment, api_server.rstrip("/"))

    def activate(self):
        # One target for ChalkPy, Compute decorators and worker launch. No secrets
        # are copied, printed or written into the receipt.
        os.environ["CHALK_ENVIRONMENT"] = self.environment
        os.environ["CHALK_ENVIRONMENT_ID"] = self.environment
        os.environ["CHALK_API_SERVER"] = self.api_server

    def client(self):
        from chalk.client import ChalkClient
        return ChalkClient(environment=self.environment, api_server=self.api_server, branch=None)


def read_project():
    if not RECEIPT.exists():
        raise ValueError("Prepare the MicroLens datasets first; see README.md step 1")
    value = json.loads(RECEIPT.read_text())
    if value.get("format_version") != 1:
        raise ValueError("Unsupported local project receipt")
    target = Target(**value["target"])
    for key in ("CHALK_ENVIRONMENT", "CHALK_ENVIRONMENT_ID"):
        if os.environ.get(key) and os.environ[key] != target.environment:
            raise ValueError(f"{key} differs from this repository's saved environment")
    if os.environ.get("CHALK_API_SERVER", target.api_server).rstrip("/") != target.api_server:
        raise ValueError("CHALK_API_SERVER differs from this repository's saved API server")
    target.activate()
    return value, target


def write_project(value):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(RECEIPT)


def new_project(target, snapshot):
    return {"format_version": 1, "target": asdict(target), "snapshot": snapshot,
            "datasets": {}, "training": {}}

