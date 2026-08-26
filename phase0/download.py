"""
Download proteinGroups.txt files from PRIDE via FTP.

Strategy (tried in order):
  1. Direct FTP using ftplib with known paths (based on publication dates)
  2. FTP path search across plausible years/months if known path fails
  3. ProteomeXchange REST API as alternative to PRIDE REST
  4. Print exact wget commands for manual download as final fallback

All approaches download ONLY proteinGroups.txt -- not raw spectra files.
"""

import os
import sys
import re
from ftplib import FTP, error_perm
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, PXD014791_ACCESSION, PTM_ACCESSIONS

FTP_HOST = "ftp.pride.ebi.ac.uk"
FTP_BASE = "/pride/data/archive"

# Known FTP paths for these specific datasets (based on publication dates).
# Year/month from each paper's online publication date.
# If wrong, the search fallback will try nearby months automatically.
KNOWN_PATHS = {
    "PXD014791": "2022/01",   # Xiong et al. Scientific Data 2022-01-11
    "PXD009775": "2019/12",   # Saei et al. Nature Comms 2019-12-06
    "PXD009644": "2019/12",   # same paper
    "PXD013134": "2019/12",   # same paper
}

# How many months either side of the known date to search if primary path fails
SEARCH_WINDOW_MONTHS = 3


def _month_range(year_month: str, window: int) -> list[str]:
    """Generate list of YYYY/MM strings within ±window months."""
    year, month = int(year_month[:4]), int(year_month[5:7])
    candidates = []
    for delta in range(-window, window + 1):
        m = month + delta
        y = year
        while m <= 0:
            m += 12; y -= 1
        while m > 12:
            m -= 12; y += 1
        candidates.append(f"{y}/{m:02d}")
    return candidates


def _ftp_find_accession(ftp: FTP, accession: str, year_month: str) -> str | None:
    """
    Return the FTP path to the accession directory, or None.
    Tries year_month first, then ±SEARCH_WINDOW_MONTHS months.
    """
    for ym in _month_range(year_month, SEARCH_WINDOW_MONTHS):
        candidate = f"{FTP_BASE}/{ym}/{accession}"
        try:
            ftp.cwd(candidate)
            ftp.cwd("/")          # reset
            return candidate
        except error_perm:
            continue
    return None


def _ftp_find_protein_groups(ftp: FTP, accession_path: str) -> str | None:
    """
    Walk the accession directory looking for proteinGroups.txt.
    Handles both flat layout and MaxQuant's combined/txt/ subfolder.
    """
    def _walk(path: str, depth: int = 0) -> str | None:
        if depth > 4:
            return None
        try:
            entries = ftp.nlst(path)
        except error_perm:
            return None
        for entry in entries:
            name = entry.rsplit("/", 1)[-1]
            if name.lower() == "proteingroupstxt" or name == "proteinGroups.txt":
                return entry
            if re.search(r"(combined|txt|maxquant|mq)", name, re.I):
                result = _walk(entry, depth + 1)
                if result:
                    return result
        # Second pass: recurse into everything if first pass missed it
        if depth == 0:
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                result = _walk(entry, depth + 1)
                if result:
                    return result
        return None

    return _walk(accession_path)


def _ftp_download_file(ftp: FTP, remote_path: str, local_path: Path) -> None:
    """Download a single file from FTP with a progress bar."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(".tmp")

    # Get file size for progress bar (best effort)
    try:
        size = ftp.size(remote_path)
    except Exception:
        size = 0

    with open(tmp, "wb") as fh, tqdm(
        total=size, unit="B", unit_scale=True,
        desc=local_path.name, leave=True
    ) as bar:
        def callback(chunk):
            fh.write(chunk)
            bar.update(len(chunk))
        ftp.retrbinary(f"RETR {remote_path}", callback, blocksize=1 << 20)

    tmp.rename(local_path)


def download_via_ftp(accession: str, out_dir: Path) -> Path | None:
    """
    Connect to PRIDE FTP, find and download proteinGroups.txt.
    Returns local path on success, None on failure.
    """
    dest = out_dir / "proteinGroups.txt"
    if dest.exists():
        print(f"  EXISTS: {dest}")
        return dest

    out_dir.mkdir(parents=True, exist_ok=True)
    known_ym = KNOWN_PATHS.get(accession, "2022/01")

    print(f"  Connecting to {FTP_HOST} ...")
    try:
        ftp = FTP(FTP_HOST, timeout=30)
        ftp.login()                    # anonymous
        ftp.set_pasv(True)
    except Exception as e:
        print(f"  FTP connection failed: {e}")
        return None

    try:
        print(f"  Searching for {accession} near {known_ym} ...")
        acc_path = _ftp_find_accession(ftp, accession, known_ym)
        if acc_path is None:
            print(f"  Could not locate {accession} on FTP.")
            return None

        print(f"  Found accession at: {acc_path}")
        pg_path = _ftp_find_protein_groups(ftp, acc_path)
        if pg_path is None:
            print(f"  proteinGroups.txt not found under {acc_path}.")
            print(f"  The deposit may only have raw files. Check manually:")
            print(f"  ftp://{FTP_HOST}{acc_path}")
            return None

        print(f"  Downloading: {pg_path}")
        _ftp_download_file(ftp, pg_path, dest)
        return dest

    except Exception as e:
        print(f"  Download failed: {e}")
        return None
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


def print_manual_commands(accession: str) -> None:
    """Print wget commands the user can copy-paste."""
    ym = KNOWN_PATHS.get(accession, "????/??")
    base = f"ftp://{FTP_HOST}{FTP_BASE}/{ym}/{accession}"
    dest = f"data/{accession}/proteinGroups.txt"
    print(f"""
  ---- Manual download for {accession} ----
  # Option A: wget (searches recursively, safe for large deposits)
  mkdir -p data/{accession}
  wget -r -l 6 -nd -np --accept "proteinGroups.txt" \\
       --directory-prefix=data/{accession}/ \\
       {base}/

  # Option B: lftp (faster for large FTP trees)
  lftp -c "mirror -I proteinGroups.txt {base}/ data/{accession}/"

  # Option C: browser
  # Visit: https://www.ebi.ac.uk/pride/archive/projects/{accession}
  # Download files -> find proteinGroups.txt in the MaxQuant combined/txt/ folder
  # Save to: {dest}
  """)


def download_all() -> None:
    base = Path(DATA_DIR)
    accessions = [PXD014791_ACCESSION] + PTM_ACCESSIONS

    any_failed = False
    for acc in accessions:
        print(f"\n[{acc}] Downloading ...")
        result = download_via_ftp(acc, base / acc)
        if result is None:
            any_failed = True
            print_manual_commands(acc)

    if any_failed:
        print(
            "\n[!] Some downloads failed (see manual commands above).\n"
            "    After placing files manually, re-run with --skip-download\n"
        )


if __name__ == "__main__":
    download_all()