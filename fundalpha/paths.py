from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "fundalpha.db"
OUT = ROOT / "out"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

for _p in (DATA, RAW, OUT, REPORTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
