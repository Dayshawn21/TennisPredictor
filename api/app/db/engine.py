from __future__ import annotations

import os
import ssl
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def normalize_asyncpg_url_and_ssl(db_url: str) -> tuple[str, dict]:
    """
    Remove sslmode/ssl from URL and convert into asyncpg connect_args.
    Works well with Neon.
    """
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    parts = urlsplit(db_url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))

    # Remove libpq-style params that asyncpg doesn't accept as kwargs
    sslmode = (q.pop("sslmode", None) or "").strip().lower()
    ssl_param = (q.pop("ssl", None) or "").strip().lower()

    cleaned_query = urlencode(q, doseq=True)
    cleaned_url = urlunsplit((parts.scheme, parts.netloc, parts.path, cleaned_query, parts.fragment))

    connect_args: dict = {}

    # sslmode handling (psycopg/libpq style)
    if sslmode:
        if sslmode == "disable":
            connect_args = {"ssl": False}
        else:
            ctx = ssl.create_default_context()
            # Neon uses valid certs; this is safe
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            connect_args = {"ssl": ctx}

    # ssl=true handling (some providers)
    if not connect_args and ssl_param:
        if ssl_param in {"0", "false", "off"}:
            connect_args = {"ssl": False}
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            connect_args = {"ssl": ctx}

    return cleaned_url, connect_args


def get_engine() -> AsyncEngine:
    db_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_ASYNC")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL (expected postgresql+asyncpg://...)")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)

    return create_async_engine(
        normalized_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
