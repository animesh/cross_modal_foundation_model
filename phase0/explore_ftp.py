"""
List the full file tree of a PRIDE FTP deposit so we can see where
proteinGroups.txt actually lives (or which zip contains it).

Usage: pipenv run python explore_ftp.py PXD014791
"""

import sys
from ftplib import FTP, error_perm

FTP_HOST = "ftp.pride.ebi.ac.uk"

KNOWN_PATHS = {
    "PXD014791": "/pride/data/archive/2021/10/PXD014791",
    "PXD009775": "/pride/data/archive/2019/11/PXD009775",
    "PXD009644": "/pride/data/archive/2019/11/PXD009644",
    "PXD013134": "/pride/data/archive/2019/11/PXD013134",
}

def list_tree(ftp: FTP, path: str, depth: int = 0, max_depth: int = 4) -> None:
    if depth > max_depth:
        return
    indent = "  " * depth
    try:
        lines = []
        ftp.dir(path, lines.append)   # use dir() not nlst() -- returns sizes too
    except error_perm as e:
        print(f"{indent}[ERR] {path}: {e}")
        return

    for line in lines:
        # Standard Unix FTP dir format:
        # drwxr-xr-x  2 user group  4096 Jan 11 2022 dirname
        # -rw-r--r--  1 user group 12345 Jan 11 2022 filename
        parts = line.split(None, 8)
        if len(parts) < 9:
            print(f"{indent}{line}")
            continue
        perms  = parts[0]
        size   = parts[4]
        name   = parts[8]
        is_dir = perms.startswith("d")

        size_str = f"{int(size)/1e6:.1f}MB" if size.isdigit() else ""
        marker   = "/" if is_dir else ""
        print(f"{indent}{name}{marker}  {size_str}")

        if is_dir and depth < max_depth:
            # Only recurse into promising subdirs
            lower = name.lower()
            if any(kw in lower for kw in
                   ("maxquant", "combined", "txt", "mq", "result", "output", "search")):
                list_tree(ftp, f"{path}/{name}", depth + 1, max_depth)
            elif depth == 0:
                # First level: always recurse to see structure
                list_tree(ftp, f"{path}/{name}", depth + 1, max_depth)


def main():
    accession = sys.argv[1] if len(sys.argv) > 1 else "PXD014791"
    base_path = KNOWN_PATHS.get(accession)
    if not base_path:
        print(f"Unknown accession: {accession}")
        print(f"Known: {list(KNOWN_PATHS.keys())}")
        sys.exit(1)

    print(f"Connecting to {FTP_HOST} ...")
    ftp = FTP(FTP_HOST, timeout=30)
    ftp.login()
    ftp.set_pasv(True)
    print(f"Connected. Listing {base_path} ...\n")
    list_tree(ftp, base_path, max_depth=3)
    ftp.quit()
    print("\nDone. Look for proteinGroups.txt or a zip/tar containing it.")
    print("Share this output to determine the right download path.")


if __name__ == "__main__":
    main()
