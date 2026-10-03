"""Service-key authentication for machine callers (the Open WebUI Pipe).

The backend has no user accounts, sessions, or cookies. The only client is
the Open WebUI Pipe, which runs server-side and authenticates with a shared
secret plus the OWUI user identity. Isolation is the network boundary (the
backend is not published) plus the service key. Corpus ownership is keyed by
`owner_ref` = the Open WebUI user id.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request

from rip_maf.core.config import settings

SERVICE_KEY_HEADER = "X-RIP-Service-Key"
SERVICE_USER_ID_HEADER = "X-RIP-User-Id"
SERVICE_USER_NAME_HEADER = "X-RIP-User-Name"


@dataclass(frozen=True)
class Principal:
    owner_ref: str
    name: str


def require_principal(request: Request) -> Principal:
    """Resolve the caller from the service-key headers, else 401."""
    key = request.headers.get(SERVICE_KEY_HEADER)
    if not key:
        raise HTTPException(status_code=401, detail="Missing service key")
    if not settings.rip_service_key or not hmac.compare_digest(
        key, settings.rip_service_key
    ):
        raise HTTPException(status_code=401, detail="Invalid service key")

    owner_ref = (request.headers.get(SERVICE_USER_ID_HEADER) or "").strip()
    if not owner_ref:
        raise HTTPException(
            status_code=401, detail=f"Missing {SERVICE_USER_ID_HEADER}"
        )
    name = (request.headers.get(SERVICE_USER_NAME_HEADER) or owner_ref).strip()
    return Principal(owner_ref=owner_ref, name=name)
