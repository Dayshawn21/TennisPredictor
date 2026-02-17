from pathlib import Path

import pandas as pd


_HERE = Path(__file__).resolve().parent
_CVS_CANDIDATES = [_HERE / "cvs", _HERE.parent / "cvs"]
for _candidate in _CVS_CANDIDATES:
	if _candidate.exists():
		CVS_DIR = _candidate
		break
else:  # pragma: no cover - setup error
	raise FileNotFoundError("Could not locate a 'cvs' directory next to this script or its parent")


def _candidate_filenames(stem: str, year: int) -> list[str]:
	s_lower = stem.lower()
	s_upper = stem.upper()
	year_str = str(year)
	return [
		f"{s_lower}_{year_str}.xlsx",
		f"{s_upper}_{year_str}.xlsx",
		f"{s_lower}_{year_str}.csv.xlsx",
		f"{s_upper}_{year_str}.csv.xlsx",
	]


def convert_excel_to_csv(stem: str, year: int = 2025) -> None:
	source = None
	for name in _candidate_filenames(stem, year):
		candidate = CVS_DIR / name
		if candidate.exists():
			source = candidate
			break

	if source is None:
		raise FileNotFoundError(
			"Expected file not found. Looked for: "
			+ ", ".join(str(CVS_DIR / name) for name in _candidate_filenames(stem, year))
		)

	if source.name.lower().endswith(".csv.xlsx"):
		target = source.with_name(source.name[:-5])  # drop only the .xlsx part
	else:
		target = source.with_suffix(".csv")
	df = pd.read_excel(source)
	df.to_csv(target, index=False, encoding="utf-8")
	print(f"Converted {source.name} -> {target.name}")


if __name__ == "__main__":
	convert_excel_to_csv("atp")
	convert_excel_to_csv("wta")
	print("✅ Conversion complete")
