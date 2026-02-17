from __future__ import annotations

from functools import lru_cache
from typing import Dict

import httpx


def _norm(s: str) -> str:
    return (s or "").strip().lower()


@lru_cache(maxsize=1)
def _cache_keys(atp: str, wta: str) -> Dict[str, str]:
    return {"ATP": atp, "WTA": wta}


async def get_atp_wta_singles_keys(client) -> Dict[str, str]:
    """
    Best-effort:
    - Try get_events() to discover keys dynamically
    - If API-Tennis returns 5xx, fallback to documented singles keys:
      ATP Singles = 265, WTA Singles = 266
    """
    try:
        data = await client.get_events()
        rows = data.get("result", []) or []

        atp_key = None
        wta_key = None

        for r in rows:
            label = _norm(r.get("event_type_type"))
            key = r.get("event_type_key")
            if key is None:
                continue
            k = str(key)

            if label == "atp singles" or ("atp" in label and "single" in label):
                atp_key = k
            if label == "wta singles" or ("wta" in label and "single" in label):
                wta_key = k

        if atp_key and wta_key:
            return _cache_keys(atp_key, wta_key)

        # If get_events worked but labels weren’t found, still fallback
        return _cache_keys("265", "266")

    except httpx.HTTPStatusError as e:
        # API-Tennis sometimes throws 500 for get_events; use documented keys
        if 500 <= e.response.status_code <= 599:
            return _cache_keys("265", "266")
        raise
