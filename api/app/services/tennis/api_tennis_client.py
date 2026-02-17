from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ApiTennisClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ["API_TENNIS_API_KEY"]
        self.base_url = (base_url or os.getenv("API_TENNIS_BASE_URL", "https://api.api-tennis.com/tennis/")).rstrip("/") + "/"
        self.timeout = timeout

    async def _get(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = dict(params or {})
        q["method"] = method
        q["APIkey"] = self.api_key

        last_exc: Exception | None = None

        logger.info("api_tennis_client.request method=%s", method)

        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.get(self.base_url, params=q)
                r.raise_for_status()
                data = r.json()

                if not isinstance(data, dict) or data.get("success") != 1:
                    raise RuntimeError(f"API-Tennis error for method={method}: {data}")

                return data

            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code
                if 500 <= status <= 599 and attempt < 3:
                    logger.warning(
                        "api_tennis_client.retry method=%s status=%s attempt=%s",
                        method,
                        status,
                        attempt,
                    )
                    await asyncio.sleep(0.5 * attempt)
                    continue
                raise

            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < 3:
                    logger.warning(
                        "api_tennis_client.retry method=%s error=%s attempt=%s",
                        method,
                        type(e).__name__,
                        attempt,
                    )
                    await asyncio.sleep(0.5 * attempt)
                    continue
                raise

        logger.error("api_tennis_client.failed method=%s error=%s", method, repr(last_exc))
        raise RuntimeError(f"API-Tennis request failed after retries: {last_exc}")

    async def get_events(self) -> Dict[str, Any]:
        return await self._get("get_events")

    async def get_fixtures(self, date_start: str, date_stop: str, event_type_key: str, timezone: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "date_start": date_start,
            "date_stop": date_stop,
            "event_type_key": event_type_key,
        }
        if timezone:
            params["timezone"] = timezone
        return await self._get("get_fixtures", params=params)
