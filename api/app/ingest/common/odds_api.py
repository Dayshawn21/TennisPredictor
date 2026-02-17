import os, requests
from requests.adapters import HTTPAdapter, Retry

BASE = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("ODDS_API_KEY")
REGIONS = os.getenv("ODDS_REGIONS", "us")
ODDS_FORMAT = os.getenv("ODDS_FORMAT", "american")

_s = requests.Session()
_r = Retry(total=4, backoff_factor=0.4, status_forcelist=[429,500,502,503,504])
_s.mount("https://", HTTPAdapter(max_retries=_r))
HEADERS = {"User-Agent": "dayshawn-sportsbot/1.0"}

def get_odds(sport_key: str, market: str = "h2h"):
    url = f"{BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": market,
        "oddsFormat": ODDS_FORMAT,
    }
    r = _s.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()
