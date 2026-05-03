"""Download Qwen2.5-1.5B-Instruct snapshot to models/ for offline use."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

from src.utils.logging import get_logger

logger = get_logger("download_model")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR = Path("models/Qwen2.5-1.5B-Instruct")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {MODEL_ID} → {TARGET_DIR}")
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(TARGET_DIR),
        ignore_patterns=["*.bin", "*.gguf"],
    )
    logger.info(f"Snapshot ready at {path}")
    cfg = TARGET_DIR / "config.json"
    assert cfg.exists(), f"Expected {cfg} after download."
    logger.info("OK.")


if __name__ == "__main__":
    main()
