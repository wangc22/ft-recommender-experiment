"""Run all three dataset preparation pipelines."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / "raw"))

from src.data import prepare_banking77, prepare_bitext, prepare_cuad
from src.utils.logging import get_logger

logger = get_logger("prepare_all")


def main():
    logger.info("=== BANKING77 ===")
    prepare_banking77.prepare()
    logger.info("=== Bitext ===")
    prepare_bitext.prepare()
    logger.info("=== CUAD ===")
    prepare_cuad.prepare()
    logger.info("All data prepared.")


if __name__ == "__main__":
    main()
