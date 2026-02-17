import os
from tennis_api_client import TennisApiClient

BASE_URL = os.environ.get("TENNIS_API_BASE_URL", "https://api.yoursite.com")
API_KEY = os.environ.get("TENNIS_API_KEY")  # optional

client = TennisApiClient(base_url=BASE_URL, api_key=API_KEY)

date = "2026-01-13"

# A) Lightweight fixtures list
fixtures = client.fixtures(date)
print("fixtures count:", len(fixtures) if isinstance(fixtures, list) else fixtures)

# B) Pick a match_key and fetch full detail
match_key = "api_tennis:d26da4facaa6960b16702d947a618fa7"
detail = client.fixture_detail(match_key)
print("detail keys:", list(detail.keys())[:20])

# C) Bulk Elo lookup for debugging/caching
elos = client.players_elo("WTA", [155305, 99296])
print("elo response:", elos)
