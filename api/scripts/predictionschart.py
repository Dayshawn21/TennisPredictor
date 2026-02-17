import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def safe_get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


def match_label(item: Dict[str, Any]) -> str:
    p1 = item.get("p1_name", "P1")
    p2 = item.get("p2_name", "P2")
    tour = item.get("tour", "")
    rnd = item.get("round", "")
    return f"{p1} vs {p2} ({tour} {rnd})".strip()


def extract_row(item: Dict[str, Any]) -> Dict[str, Any]:
    inputs = item.get("inputs", {}) or {}
    sim = inputs.get("totals_sets_sim", {}) or {}
    props = inputs.get("projected_props", {}) or {}
    aces = props.get("aces", {}) or {}
    bp = props.get("break_points", {}) or {}

    exp_games = to_float(sim.get("expected_total_games"))
    if exp_games is None:
        exp_games = to_float(item.get("projected_total_games"))

    # Aces
    total_aces = to_float(aces.get("total_expected"))
    if total_aces is None:
        p1a = to_float(aces.get("p1_expected")) or 0.0
        p2a = to_float(aces.get("p2_expected")) or 0.0
        total_aces = p1a + p2a

    # Breaks + BP created
    total_breaks = (to_float(bp.get("p1_breaks")) or 0.0) + (to_float(bp.get("p2_breaks")) or 0.0)
    total_bp_created = (to_float(bp.get("p1_created")) or 0.0) + (to_float(bp.get("p2_created")) or 0.0)

    return {
        "match_id": item.get("match_id"),
        "label": match_label(item),
        "surface": item.get("surface"),
        "expected_games": exp_games,
        "total_aces": total_aces,
        "total_breaks": total_breaks,
        "total_bp_created": total_bp_created,
    }


def top_n(rows: List[Dict[str, Any]], key: str, n: int = 10) -> List[Dict[str, Any]]:
    # Sort descending, keep rows with numeric values
    valid = [r for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) is not None]
    return sorted(valid, key=lambda r: r[key], reverse=True)[:n]


def barh_top(rows: List[Dict[str, Any]], key: str, title: str, outpath: Path):
    rows = list(reversed(rows))  # reverse so biggest is at top when inverted
    labels = [r["label"] for r in rows]
    values = [r[key] for r in rows]

    plt.figure(figsize=(12, 7))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(key.replace("_", " ").title())
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def scatter_aces_vs_games(rows: List[Dict[str, Any]], outpath: Path):
    pts = [r for r in rows if isinstance(r.get("expected_games"), (int, float)) and isinstance(r.get("total_aces"), (int, float))]
    x = [r["expected_games"] for r in pts]
    y = [r["total_aces"] for r in pts]

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y)
    plt.title("Total Expected Aces vs Expected Total Games")
    plt.xlabel("Expected Total Games")
    plt.ylabel("Total Expected Aces")

    # annotate lightly (first 12 to avoid clutter)
    for r in pts[:12]:
        plt.annotate(r["label"].split("(")[0].strip(), (r["expected_games"], r["total_aces"]), fontsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    api_dir = Path(__file__).resolve().parents[1]  # .../api
    json_path = api_dir / "app" / "data" / "predictions.json"
    out_dir = api_dir / "app" / "data" / "charts"

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find predictions.json at: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = data.get("items", []) or []

    rows = [extract_row(it) for it in items]

    # 1) Top 10 total expected aces
    aces_top = top_n(rows, "total_aces", 10)
    barh_top(
        aces_top,
        "total_aces",
        "Total Expected Aces (Top 10 Matches)",
        out_dir / "top10_total_aces.png",
    )

    # 2) Top 10 breaks + Top 10 BP created
    breaks_top = top_n(rows, "total_breaks", 10)
    barh_top(
        breaks_top,
        "total_breaks",
        "Total Expected Breaks (Top 10 Matches)",
        out_dir / "top10_total_breaks.png",
    )

    bp_top = top_n(rows, "total_bp_created", 10)
    barh_top(
        bp_top,
        "total_bp_created",
        "Total Break Points Created (Top 10 Matches)",
        out_dir / "top10_total_bp_created.png",
    )

    # 3) Scatter
    scatter_aces_vs_games(rows, out_dir / "aces_vs_expected_games.png")

    print(f"Saved charts to: {out_dir}")


if __name__ == "__main__":
    main()
