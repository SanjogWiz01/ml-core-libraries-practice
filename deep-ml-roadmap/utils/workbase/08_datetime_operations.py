from datetime import datetime, timedelta, timezone
now = datetime.now(timezone.utc)
print(now)
print(now + timedelta(days=7))
print(datetime.strptime("2026-09-15", "%Y-%m-%d"))
print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
