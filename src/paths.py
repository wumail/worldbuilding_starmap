"""Repository layout: generators live in src/, catalogs in output/, priors in data/."""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
WEB_DIR = ROOT / "web"
DESIGN_DIR = ROOT / "design"
