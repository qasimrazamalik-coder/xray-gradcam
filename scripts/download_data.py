"""Download the Kaggle 'Chest X-Ray Images (Pneumonia)' dataset (Paul Mooney's upload).

Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
The images derive from paediatric patients (ages one to five) imaged at Guangzhou Women and
Children's Medical Center; the underlying study is Kermany et al., Cell, 2018.

Licence: not asserted here. Read the licence and terms on the Kaggle dataset page before
using the data or sharing anything derived from it.

Requires Kaggle API credentials (https://www.kaggle.com/docs/api): either ~/.kaggle/kaggle.json
or the KAGGLE_USERNAME / KAGGLE_KEY environment variables, and `pip install kaggle`.
The archive is a few GB on disk once extracted (about 5,800 JPEGs).

Usage: python scripts/download_data.py [--out data]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SLUG = "paultimothymooney/chest-xray-pneumonia"


def main(out: Path) -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "The `kaggle` CLI is not installed. Run `pip install kaggle` and set up credentials."
        )
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(out), "--unzip"], check=True
    )
    root = next(
        (
            p
            for p in (out / "chest_xray", out / "chest_xray" / "chest_xray")
            if (p / "train").is_dir()
        ),
        None,
    )
    if root is None:
        sys.exit(
            f"Download finished but no train/ folder found under {out}. Check the archive layout."
        )
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            n = len(list((root / split / cls).glob("*.jp*g")))
            print(f"{split}/{cls}: {n} images")
    print(f"Dataset ready. Set data.root to '{out / 'chest_xray'}' (default in configs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    main(parser.parse_args().out)
