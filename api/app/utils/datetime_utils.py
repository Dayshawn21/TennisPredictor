from datetime import datetime
import pytz

CENTRAL_TZ = pytz.timezone("US/Central")

def match_date_central(start_ts: int):
    """
    Convert a UTC epoch timestamp (seconds) to Central Time match date.
    """
    dt_utc = datetime.utcfromtimestamp(start_ts).replace(tzinfo=pytz.UTC)
    dt_central = dt_utc.astimezone(CENTRAL_TZ)
    return dt_central.date()
