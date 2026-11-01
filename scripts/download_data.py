"""Download UCI #350 and reproducibly retain financial features only.

Usage: python scripts/download_data.py [--archive path/to/uci.zip]
The app itself uses the bundled compressed CSV and never downloads on startup.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from credit_risk.data import EXCLUDED_FEATURES, FINANCIAL_FEATURES, TARGET

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"


def prepare_dataset(archive: bytes, output: Path) -> dict:
    """Validate the source checksum; emit financial CSV plus provenance JSON."""
    source_hash = hashlib.sha256(archive).hexdigest()
    if source_hash != ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum changed. Review the upstream file before updating the pinned checksum.")
    with ZipFile(io.BytesIO(archive)) as zipped:
        names = [name for name in zipped.namelist() if name.lower().endswith(".xls")]
        if len(names) != 1:
            raise ValueError("Expected exactly one UCI .xls workbook.")
        source = pd.read_excel(io.BytesIO(zipped.read(names[0])), header=1, engine="xlrd")
    source.columns = [str(c).strip() for c in source.columns]
    target_name = "default payment next month"
    if len(source) != 30000 or not set(FINANCIAL_FEATURES + [target_name]).issubset(source.columns):
        raise ValueError("The UCI workbook has an unexpected shape or schema.")
    financial = source[FINANCIAL_FEATURES + [target_name]].rename(columns={target_name: TARGET})
    if financial.isna().any().any() or not financial[TARGET].isin([0, 1]).all():
        raise ValueError("Unexpected missing financial values or nonbinary target.")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = financial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    # Empty gzip filename and fixed mtime make the derived file reproducible.
    with output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(payload)
    metadata = {
        "dataset": "Default of Credit Card Clients", "uci_id": 350,
        "source_url": UCI_URL,
        "source_page": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": source_hash,
        "csv_gzip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(financial), "financial_features": FINANCIAL_FEATURES,
        "removed_columns": EXCLUDED_FEATURES,
        "target": TARGET, "positive_class": "default payment next month = 1",
        "positive_count": int(financial[TARGET].sum()),
        "missing_financial_cells": int(financial.isna().sum().sum()),
        "observations": "Taiwan credit-card clients; six monthly histories April-September 2005; amounts in NT dollars.",
        "transformations": "Original row order retained; removed ID/SEX/EDUCATION/MARRIAGE/AGE; renamed target to default; no row sampling or target alteration.",
        "status_codes": "Observed -2, -1, and 0 are retained as separate categories. The source clearly describes -1 and positive delays but does not fully resolve -2 and 0 semantics.",
    }
    output.with_name("provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already-downloaded original UCI ZIP.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "credit_default.csv.gz")
    args = parser.parse_args()
    if args.archive:
        archive = args.archive.read_bytes()
    else:
        request = Request(UCI_URL, headers={"User-Agent": "EducationalCreditRiskDashboard/1.0"})
        with urlopen(request, timeout=60) as response:
            archive = response.read()
    metadata = prepare_dataset(archive, args.output)
    print(f"Prepared {metadata['rows']:,} rows, 19 financial features: {args.output}")
    print(f"SHA256 {metadata['csv_gzip_sha256']}")


if __name__ == "__main__":
    main()
