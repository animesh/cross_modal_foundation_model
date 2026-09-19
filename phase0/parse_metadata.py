"""
Download Metadata-Table.xlsx from PRIDE PXD014791 FTP and display its contents.
This file maps each gel/experiment to the iPSC-CM donor line.

Usage: pipenv run python parse_metadata.py
"""

import sys
from ftplib import FTP
from pathlib import Path

FTP_HOST  = "ftp.pride.ebi.ac.uk"
META_PATH = "/pride/data/archive/2021/10/PXD014791/Metadata-Table.xlsx"
OUT_PATH  = Path("data/PXD014791/Metadata-Table.xlsx")


def download_metadata() -> Path:
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH}")
        return OUT_PATH

    print(f"Connecting to {FTP_HOST} ...")
    ftp = FTP(FTP_HOST, timeout=30)
    ftp.login()
    ftp.set_pasv(True)
    ftp.voidcmd("TYPE I")

    size = ftp.size(META_PATH)
    print(f"Downloading Metadata-Table.xlsx ({size} bytes) ...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        ftp.retrbinary(f"RETR {META_PATH}", fh.write)

    ftp.quit()
    print(f"Saved: {OUT_PATH}")
    return OUT_PATH


def display_metadata(path: Path) -> None:
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not installed -- installing ...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl",
                        "--break-system-packages", "-q"])

    import pandas as pd

    print(f"\n=== {path.name} ===")
    xl = pd.ExcelFile(path)
    print(f"Sheets: {xl.sheet_names}")

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        print(f"\n--- Sheet: {sheet!r} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(20).to_string())
        print()


if __name__ == "__main__":
    path = download_metadata()
    display_metadata(path)
    print(
        "\nShare the output above to determine:\n"
        "  1. Which column contains the sample/file name\n"
        "  2. Which column contains the iPSC-CM donor line\n"
        "Then parse_lfq.py will apply the mapping automatically."
    )
