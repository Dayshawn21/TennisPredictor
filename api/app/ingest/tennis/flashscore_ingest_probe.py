"""
flashscore_ingest_probe.py

Test-only Flashscore tennis ingest probe (no DB writes).

Modes:
- desktop: Playwright-rendered odds table (includes spread, moneyline, total)
- mobile: m.flashscoreusa HTML odds page (moneyline-focused fallback)

Usage:
  py -3 api/app/ingest/tennis/flashscore_ingest_probe.py
  py -3 api/app/ingest/tennis/flashscore_ingest_probe.py --mode desktop --out api/app/data/flashscore_probe.json
  py -3 api/app/ingest/tennis/flashscore_ingest_probe.py --mode mobile --day tomorrow
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


DESKTOP_URL = "https://www.flashscoreusa.com/tennis/"
MOBILE_BASE_URL = "https://m.flashscoreusa.com/tennis/"
DAY_TO_QUERY = {
    "today": "d=0&s=5",
    "yesterday": "d=-1&s=5",
    "tomorrow": "d=1&s=5",
}


@dataclass
class ProbeMatch:
    source: str
    mode: str
    match_external_id: Optional[str]
    tournament: Optional[str]
    start_time_local: Optional[str]
    p1_name: Optional[str]
    p2_name: Optional[str]
    p1_score: Optional[str]
    p2_score: Optional[str]
    score_text: Optional[str]
    status_text: Optional[str]
    spread_p1_line: Optional[str]
    spread_p1_odds_american: Optional[int]
    moneyline_p1_american: Optional[int]
    total_over_line: Optional[str]
    total_over_odds_american: Optional[int]
    spread_p2_line: Optional[str]
    spread_p2_odds_american: Optional[int]
    moneyline_p2_american: Optional[int]
    total_under_line: Optional[str]
    total_under_odds_american: Optional[int]
    moneyline_decimal_p1: Optional[float]
    moneyline_decimal_p2: Optional[float]
    raw_odds_cells: list[str]
    raw_odds_cells_list: list[str]
    raw_odds_cells_detail: list[str]
    detail_url: Optional[str]


def _clean_html_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ")
    s = s.replace("&amp;", "&")
    return " ".join(s.split()).strip()


def _extract_external_id(raw: str) -> Optional[str]:
    if not raw:
        return None
    if raw.startswith("g_"):
        return raw.split("_")[-1]
    if "/game/" in raw:
        x = raw.split("/game/", 1)[1]
        return x.split("/", 1)[0]
    return raw


def _to_float_or_none(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _american_from_decimal(v: Optional[float]) -> Optional[int]:
    if v is None or v <= 1.0:
        return None
    if v >= 2.0:
        return int(round((v - 1.0) * 100))
    return int(round(-100.0 / (v - 1.0)))


def _to_american_or_none(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    t = str(v).strip().upper()
    if not t or t == "-":
        return None
    if t == "EV":
        return 100
    t = t.replace("+", "")
    try:
        n = int(float(t))
        return n
    except Exception:
        return None


def _parse_moneyline_row(text: str) -> tuple[Optional[int], Optional[int]]:
    toks = text.split()
    if len(toks) < 2:
        return None, None
    return _to_american_or_none(toks[-2]), _to_american_or_none(toks[-1])


def _parse_total_row(text: str) -> tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    toks = text.split()
    if len(toks) < 3:
        return None, None, None, None
    line = toks[0]
    over = _to_american_or_none(toks[1])
    under = _to_american_or_none(toks[2])
    return f"Ov {line}", over, f"Un {line}", under


def _parse_spread_row(text: str) -> tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    toks = text.split()
    if len(toks) < 3:
        return None, None, None, None
    pair = toks[0]
    home = _to_american_or_none(toks[1])
    away = _to_american_or_none(toks[2])
    if "/" in pair:
        left, right = pair.split("/", 1)
        return left, home, right, away
    return pair, home, None, away


def _normalize_score_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    t = str(v).strip()
    if not t or t in {"-", "--"}:
        return None
    return t


def _is_allowed_tournament(tournament: str, tour_filter: set[str], singles_only: bool) -> bool:
    t = (tournament or "").upper()
    if singles_only and "SINGLES" not in t:
        return False
    if not tour_filter:
        return True
    return any(tag in t for tag in tour_filter)


def _match_from_odds_cells(
    *,
    mode: str,
    match_external_id: Optional[str],
    tournament: Optional[str],
    start_time_local: Optional[str],
    p1_name: Optional[str],
    p2_name: Optional[str],
    p1_score: Optional[str],
    p2_score: Optional[str],
    status_text: Optional[str],
    odds_cells: list[str],
    moneyline_decimal_pair: tuple[Optional[float], Optional[float]] = (None, None),
) -> ProbeMatch:
    cells = odds_cells + [""] * (10 - len(odds_cells))
    # Expected rendered order:
    # 0 spread p1 line, 1 spread p1 odds, 2 ml p1,
    # 3 total over line, 4 total over odds,
    # 5 spread p2 line, 6 spread p2 odds, 7 ml p2,
    # 8 total under line, 9 total under odds.
    return ProbeMatch(
        source="flashscore",
        mode=mode,
        match_external_id=match_external_id,
        tournament=tournament,
        start_time_local=start_time_local,
        p1_name=p1_name,
        p2_name=p2_name,
        p1_score=_normalize_score_value(p1_score),
        p2_score=_normalize_score_value(p2_score),
        score_text=(
            f"{_normalize_score_value(p1_score)}-{_normalize_score_value(p2_score)}"
            if _normalize_score_value(p1_score) and _normalize_score_value(p2_score)
            else None
        ),
        status_text=status_text,
        spread_p1_line=(cells[0] or None),
        spread_p1_odds_american=_to_american_or_none(cells[1]),
        moneyline_p1_american=_to_american_or_none(cells[2]),
        total_over_line=(cells[3] or None),
        total_over_odds_american=_to_american_or_none(cells[4]),
        spread_p2_line=(cells[5] or None),
        spread_p2_odds_american=_to_american_or_none(cells[6]),
        moneyline_p2_american=_to_american_or_none(cells[7]),
        total_under_line=(cells[8] or None),
        total_under_odds_american=_to_american_or_none(cells[9]),
        moneyline_decimal_p1=moneyline_decimal_pair[0],
        moneyline_decimal_p2=moneyline_decimal_pair[1],
        raw_odds_cells=odds_cells,
        raw_odds_cells_list=odds_cells,
        raw_odds_cells_detail=[],
        detail_url=None,
    )


async def _fetch_detail_main_markets(page, detail_url: str, mid: str) -> dict[str, Any]:
    base = detail_url.split("?", 1)[0].rstrip("/")
    urls = {
        "moneyline": f"{base}/odds/money-line/?mid={mid}",
        "total": f"{base}/odds/total/?mid={mid}",
        "spread": f"{base}/odds/spread/?mid={mid}",
    }

    out: dict[str, Any] = {}
    for key, url in urls.items():
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=90000)
            await page.wait_for_timeout(2500)
            rows = []
            for e in await page.query_selector_all(".ui-table__body .ui-table__row"):
                try:
                    t = " ".join((await e.inner_text()).split()).strip()
                except Exception:
                    continue
                if t:
                    rows.append(t)
            out[key] = rows
            if key == "moneyline":
                try:
                    st = await page.query_selector(".duelParticipant__startTime")
                    if st:
                        st_text = " ".join((await st.inner_text()).split()).strip()
                        if st_text:
                            out["start_time_local"] = st_text
                except Exception:
                    pass
                try:
                    sc = await page.query_selector(".detailScore__wrapper")
                    if sc:
                        sc_text = " ".join((await sc.inner_text()).split()).strip()
                        if sc_text:
                            m = re.search(r"(\d+)\s*-\s*(\d+)", sc_text)
                            if m:
                                out["p1_score"] = _normalize_score_value(m.group(1))
                                out["p2_score"] = _normalize_score_value(m.group(2))
                except Exception:
                    pass
        except Exception:
            out[key] = []
    return out


async def _probe_desktop(
    day: str,
    tour_filter: set[str],
    singles_only: bool,
    detail_odds: bool,
    detail_all: bool,
    detail_limit: int,
) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(DESKTOP_URL, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_timeout(5500)

        # Click ODDS tab/filter.
        for label in ("ODDS", "Odds"):
            try:
                node = page.get_by_text(label, exact=True).first
                if await node.is_visible(timeout=1500):
                    await node.click(timeout=3000)
                    break
            except Exception:
                continue
        await page.wait_for_timeout(3000)

        rows = await page.query_selector_all("div.event__match")
        items: list[ProbeMatch] = []
        for row in rows:
            row_id = (await row.get_attribute("id")) or ""
            match_external_id = _extract_external_id(row_id)

            names = []
            for e in await row.query_selector_all(".event__participant"):
                t = " ".join((await e.inner_text()).split()).strip()
                if t:
                    names.append(t)
            p1_name = names[0] if len(names) > 0 else None
            p2_name = names[1] if len(names) > 1 else None
            p1_score = None
            p2_score = None
            try:
                hs = await row.query_selector(".event__score--home")
                if hs:
                    t = " ".join((await hs.inner_text()).split()).strip()
                    p1_score = t or None
                as_ = await row.query_selector(".event__score--away")
                if as_:
                    t = " ".join((await as_.inner_text()).split()).strip()
                    p2_score = t or None
            except Exception:
                p1_score = None
                p2_score = None

            status_text = None
            stage = await row.query_selector(".event__stage--block")
            if stage:
                t = " ".join((await stage.inner_text()).split()).strip()
                status_text = t or None

            time_el = await row.query_selector(".event__time")
            start_time_local = None
            if time_el:
                t = " ".join((await time_el.inner_text()).split()).strip()
                start_time_local = t or None

            tournament = None
            category = None
            try:
                header_data = await row.evaluate(
                    """
                    (n) => {
                      let p = n.previousElementSibling;
                      while (p && !(p.className || '').includes('headerLeague__wrapper')) {
                        p = p.previousElementSibling;
                      }
                      if (!p) return null;
                      const titleEl = p.querySelector('.headerLeague__title-text') || p.querySelector('.headerLeague__title');
                      const catEl = p.querySelector('.headerLeague__category-text') || p.querySelector('.headerLeague__category');
                      const title = titleEl ? (titleEl.textContent || '').trim() : '';
                      let category = catEl ? (catEl.textContent || '').trim() : '';
                      category = category.replace(/:$/, '').trim();
                      return {title, category};
                    }
                    """
                )
                if header_data:
                    title_txt = (header_data.get("title") or "").strip()
                    cat_txt = (header_data.get("category") or "").strip()
                    category = cat_txt or None
                    if title_txt and cat_txt:
                        tournament = f"{cat_txt}: {title_txt}"
                    elif title_txt:
                        tournament = title_txt
            except Exception:
                tournament = None
                category = None

            filter_text = category or tournament or ""
            if not _is_allowed_tournament(filter_text, tour_filter, singles_only):
                continue

            odds_cells = []
            for e in await row.query_selector_all(".odds__odd"):
                t = " ".join((await e.inner_text()).split()).strip()
                if t:
                    odds_cells.append(t)

            m = _match_from_odds_cells(
                    mode="desktop",
                    match_external_id=match_external_id,
                    tournament=tournament,
                    start_time_local=start_time_local,
                    p1_name=p1_name,
                    p2_name=p2_name,
                    p1_score=p1_score,
                    p2_score=p2_score,
                    status_text=status_text,
                    odds_cells=odds_cells,
                )
            link = await row.query_selector("a.eventRowLink")
            href = await link.get_attribute("href") if link else None
            m.detail_url = href
            items.append(m)

        if detail_odds:
            patched = 0
            detail_page = await context.new_page()
            for m in items:
                if detail_limit > 0 and patched >= detail_limit:
                    break
                if not m.match_external_id or not m.detail_url:
                    continue
                missing_main = (
                    m.moneyline_p1_american is None
                    or m.moneyline_p2_american is None
                    or m.spread_p1_line in (None, "-")
                    or m.spread_p2_line in (None, "-")
                    or m.total_over_line in (None, "-")
                    or m.total_under_line in (None, "-")
                )
                if (not detail_all) and (not missing_main):
                    continue

                rows_by_market = await _fetch_detail_main_markets(detail_page, m.detail_url, m.match_external_id)
                if not m.start_time_local:
                    st = rows_by_market.get("start_time_local")
                    if isinstance(st, str) and st.strip():
                        m.start_time_local = st.strip()
                if not m.p1_score or not m.p2_score:
                    p1s = rows_by_market.get("p1_score")
                    p2s = rows_by_market.get("p2_score")
                    if isinstance(p1s, str) and p1s.strip():
                        m.p1_score = _normalize_score_value(p1s.strip())
                    if isinstance(p2s, str) and p2s.strip():
                        m.p2_score = _normalize_score_value(p2s.strip())
                    if m.p1_score and m.p2_score:
                        m.score_text = f"{m.p1_score}-{m.p2_score}"
                ml_rows = rows_by_market.get("moneyline") or []
                tt_rows = rows_by_market.get("total") or []
                sp_rows = rows_by_market.get("spread") or []
                detail_cells = ["", "", "", "", "", "", "", "", "", ""]

                if ml_rows:
                    p1ml, p2ml = _parse_moneyline_row(ml_rows[0])
                    if p1ml is not None:
                        m.moneyline_p1_american = p1ml
                        detail_cells[2] = f"{p1ml:+d}"
                    if p2ml is not None:
                        m.moneyline_p2_american = p2ml
                        detail_cells[7] = f"{p2ml:+d}"
                if tt_rows:
                    ov_line, ov_odds, un_line, un_odds = _parse_total_row(tt_rows[0])
                    if ov_line:
                        m.total_over_line = ov_line
                        detail_cells[3] = ov_line
                    if un_line:
                        m.total_under_line = un_line
                        detail_cells[8] = un_line
                    if ov_odds is not None:
                        m.total_over_odds_american = ov_odds
                        detail_cells[4] = f"{ov_odds:+d}"
                    if un_odds is not None:
                        m.total_under_odds_american = un_odds
                        detail_cells[9] = f"{un_odds:+d}"
                if sp_rows:
                    sp1, sp1o, sp2, sp2o = _parse_spread_row(sp_rows[0])
                    if sp1:
                        m.spread_p1_line = sp1
                        detail_cells[0] = sp1
                    if sp2:
                        m.spread_p2_line = sp2
                        detail_cells[5] = sp2
                    if sp1o is not None:
                        m.spread_p1_odds_american = sp1o
                        detail_cells[1] = f"{sp1o:+d}"
                    if sp2o is not None:
                        m.spread_p2_odds_american = sp2o
                        detail_cells[6] = f"{sp2o:+d}"
                if any(x for x in detail_cells):
                    m.raw_odds_cells_detail = detail_cells
                    m.raw_odds_cells = detail_cells
                patched += 1
            await detail_page.close()

        await browser.close()

    return {
        "source": "flashscore_probe",
        "mode": "desktop",
        "url": DESKTOP_URL,
        "day": day,
        "count": len(items),
        "items": [asdict(m) for m in items],
    }


def _fetch_mobile_html(day: str) -> tuple[str, str]:
    query = DAY_TO_QUERY[day]
    url = f"{MOBILE_BASE_URL}?{query}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", "ignore")
    return url, html


def _extract_mobile_score_data_block(html: str) -> str:
    m = re.search(r'<div id="score-data">(.*?)</div>', html, re.S | re.I)
    return m.group(1) if m else ""


def _probe_mobile(day: str, tour_filter: set[str], singles_only: bool) -> dict[str, Any]:
    url, html = _fetch_mobile_html(day)
    block = _extract_mobile_score_data_block(html)
    items: list[ProbeMatch] = []
    current_tournament: Optional[str] = None

    chunks = re.split(r"(<h4>.*?</h4>)", block, flags=re.S | re.I)
    for chunk in chunks:
        h = re.search(r"<h4>(.*?)</h4>", chunk, re.S | re.I)
        if h:
            current_tournament = _clean_html_text(h.group(1))
            continue

        line_pat = re.compile(
            r"<span>(?P<time>[^<]+)</span>\s*"
            r"(?P<names>.*?)\s*"
            r'<a href="(?P<href>/game/[^"]+)"[^>]*>(?P<status>[^<]*)</a>'
            r"(?:\s*<span class=\"mobi-odds\">.*?"
            r">(?P<odd1>\d+(?:\.\d+)?)</a>\s*\|\s*"
            r".*?>(?P<odd2>\d+(?:\.\d+)?)</a>.*?</span>)?",
            re.S | re.I,
        )
        for m in line_pat.finditer(chunk):
            if not current_tournament:
                continue
            if not _is_allowed_tournament(current_tournament, tour_filter, singles_only):
                continue

            names_text = _clean_html_text(m.group("names"))
            p1 = None
            p2 = None
            if " - " in names_text:
                p1, p2 = names_text.rsplit(" - ", 1)
            else:
                p1 = names_text

            odd1 = _to_float_or_none((m.group("odd1") or "").strip())
            odd2 = _to_float_or_none((m.group("odd2") or "").strip())
            ml1 = _american_from_decimal(odd1)
            ml2 = _american_from_decimal(odd2)
            cells = [
                "",
                "",
                f"{ml1:+d}" if ml1 is not None else "",
                "",
                "",
                "",
                "",
                f"{ml2:+d}" if ml2 is not None else "",
                "",
                "",
            ]
            items.append(
                _match_from_odds_cells(
                    mode="mobile",
                    match_external_id=_extract_external_id(m.group("href") or ""),
                    tournament=current_tournament,
                    start_time_local=_clean_html_text(m.group("time") or "") or None,
                    p1_name=(p1 or "").strip() or None,
                    p2_name=(p2 or "").strip() or None,
                    p1_score=None,
                    p2_score=None,
                    status_text=_clean_html_text(m.group("status") or "") or None,
                    odds_cells=cells,
                    moneyline_decimal_pair=(odd1, odd2),
                )
            )

    return {
        "source": "flashscore_probe",
        "mode": "mobile",
        "url": url,
        "day": day,
        "count": len(items),
        "items": [asdict(m) for m in items],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["desktop", "mobile"], default="desktop")
    parser.add_argument("--day", choices=["today", "yesterday", "tomorrow"], default="today")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON file")
    parser.add_argument(
        "--tour-filter",
        type=str,
        default="ATP,WTA",
        help="Comma separated tournament tags to keep (example: ATP,WTA). Empty keeps all.",
    )
    parser.add_argument(
        "--include-doubles",
        action="store_true",
        help="Include doubles rows. Default is singles only.",
    )
    parser.add_argument(
        "--detail-odds",
        action="store_true",
        help="For desktop mode: open each matchup odds page and parse MONEYLINE/TOTAL/SPREAD tabs.",
    )
    parser.add_argument(
        "--detail-all",
        action="store_true",
        help="With --detail-odds, deep-parse all rows (default deep-parses only rows with missing main markets).",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=15,
        help="Max rows to deep-parse when --detail-odds is used. 0 means no limit.",
    )
    args = parser.parse_args()

    tour_filter = {x.strip().upper() for x in (args.tour_filter or "").split(",") if x.strip()}
    singles_only = not args.include_doubles

    if args.mode == "desktop":
        payload = asyncio.run(
            _probe_desktop(
                args.day,
                tour_filter,
                singles_only,
                detail_odds=args.detail_odds,
                detail_all=args.detail_all,
                detail_limit=max(0, int(args.detail_limit)),
            )
        )
    else:
        payload = _probe_mobile(args.day, tour_filter, singles_only)

    text = json.dumps(payload, indent=2, ensure_ascii=True)
    print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
