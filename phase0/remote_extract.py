"""
Extract proteinGroups.txt from PRIDE Search.zip using FTP REST (seek) + RETR.

PRIDE HTTPS does not honor Range requests despite returning Content-Length.
FTP protocol supports the REST command which sets a restart offset before RETR,
giving the same functionality. We read exactly as many bytes as we need and
close the data socket early.

Approach (4 FTP partial reads, ~100-200 MB total):
  1. SIZE command       -> total zip size
  2. REST + RETR tail   -> last 64KB to find EOCD / ZIP64 locator
  3. REST + RETR cd     -> central directory to locate target file
  4. REST + RETR data   -> compressed data of proteinGroups.txt

Usage:
    pipenv run python remote_extract.py PXD014791
    pipenv run python remote_extract.py PXD009775   # after running explore_ftp.py first
"""

import struct
import sys
import zlib
from ftplib import FTP, error_reply
from pathlib import Path
from tqdm import tqdm

FTP_HOST = "ftp.pride.ebi.ac.uk"

FTP_PATHS = {
    "PXD014791": "/pride/data/archive/2021/10/PXD014791/Search.zip",
    "PXD009775": "/pride/data/archive/2019/11/PXD009775/Search.zip",
    "PXD009644": "/pride/data/archive/2019/11/PXD009644/Search.zip",
    "PXD013134": "/pride/data/archive/2019/11/PXD013134/Search.zip",
}

OUTPUT_DIR = Path("data")
TARGET_FILE = "proteinGroups.txt"


# ---------------------------------------------------------------------------
# FTP partial-read helper
# ---------------------------------------------------------------------------

def _ftp_connect() -> FTP:
    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login()
    ftp.set_pasv(True)
    ftp.voidcmd("TYPE I")   # binary mode
    return ftp


def ftp_read_range(ftp_path: str, start: int, length: int,
                   show_progress: bool = False) -> bytes:
    """
    Download exactly `length` bytes from `ftp_path` beginning at byte `start`.
    Uses FTP REST to seek, RETR to start transfer, closes data socket early.
    Retries on empty recv -- first packet often arrives late for small reads.
    """
    if length == 0:
        return b''

    ftp = _ftp_connect()
    try:
        ftp.sendcmd(f"REST {start}")

        buf = bytearray()
        with ftp.transfercmd(f"RETR {ftp_path}") as sock:
            sock.settimeout(120)
            remaining = length
            empty_streak = 0
            bar = tqdm(total=length, unit="B", unit_scale=True,
                       disable=not show_progress, leave=False)
            while remaining > 0:
                chunk = sock.recv(min(remaining, 1 << 20))
                if not chunk:
                    empty_streak += 1
                    if empty_streak > 20:   # give up after 10s
                        break
                    import time; time.sleep(0.5)
                    continue
                empty_streak = 0
                buf.extend(chunk)
                remaining -= len(chunk)
                bar.update(len(chunk))
            bar.close()

        try:
            ftp.voidresp()
        except Exception:
            pass

    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    return bytes(buf)


def ftp_file_size(ftp_path: str) -> int:
    ftp = _ftp_connect()
    try:
        size = ftp.size(ftp_path)
    finally:
        try: ftp.quit()
        except Exception: pass
    return size


# ---------------------------------------------------------------------------
# ZIP / ZIP64 central directory parsing (unchanged from http version)
# ---------------------------------------------------------------------------

EOCD_SIG       = b"PK\x05\x06"
ZIP64_EOCD_SIG = b"PK\x06\x06"
ZIP64_EOCD_LOC = b"PK\x06\x07"
CD_ENTRY_SIG   = b"PK\x01\x02"
LOCAL_ENTRY_SIG= b"PK\x03\x04"


def _find_eocd(tail: bytes) -> int:
    pos = tail.rfind(EOCD_SIG)
    if pos == -1:
        raise ValueError("EOCD signature not found in last 64KB")
    return pos


def _parse_eocd(tail: bytes, pos: int) -> tuple[int, int]:
    cd_size   = struct.unpack_from("<I", tail, pos + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, pos + 16)[0]
    return cd_offset, cd_size


def _parse_zip64_eocd(ftp_path: str, tail: bytes, file_size: int) -> tuple[int, int]:
    loc_pos = tail.rfind(ZIP64_EOCD_LOC)
    if loc_pos == -1:
        raise ValueError("ZIP64 EOCD Locator not found")
    zip64_eocd_abs = struct.unpack_from("<Q", tail, loc_pos + 8)[0]
    z64 = ftp_read_range(ftp_path, zip64_eocd_abs, 56)
    if z64[:4] != ZIP64_EOCD_SIG:
        raise ValueError(f"ZIP64 EOCD sig mismatch: {z64[:4]!r}")
    cd_size   = struct.unpack_from("<Q", z64, 40)[0]
    cd_offset = struct.unpack_from("<Q", z64, 48)[0]
    return cd_offset, cd_size


def _resolve_zip64_extra(extra: bytes, c32: int, u32: int, lh32: int):
    c, u, lh = c32, u32, lh32
    pos = 0
    while pos + 4 <= len(extra):
        tag  = struct.unpack_from("<H", extra, pos)[0]
        size = struct.unpack_from("<H", extra, pos + 2)[0]
        data = extra[pos+4:pos+4+size]
        if tag == 0x0001:
            off = 0
            if u32  == 0xFFFFFFFF and off+8 <= len(data): u  = struct.unpack_from("<Q",data,off)[0]; off+=8
            if c32  == 0xFFFFFFFF and off+8 <= len(data): c  = struct.unpack_from("<Q",data,off)[0]; off+=8
            if lh32 == 0xFFFFFFFF and off+8 <= len(data): lh = struct.unpack_from("<Q",data,off)[0]
        pos += 4 + size
    return c, u, lh


def _search_cd(cd_data: bytes, target: str) -> dict | None:
    pos = 0
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos+4] != CD_ENTRY_SIG:
            break
        comp_method   = struct.unpack_from("<H", cd_data, pos+10)[0]
        comp_size_32  = struct.unpack_from("<I", cd_data, pos+20)[0]
        uncomp_size_32= struct.unpack_from("<I", cd_data, pos+24)[0]
        fname_len     = struct.unpack_from("<H", cd_data, pos+28)[0]
        extra_len     = struct.unpack_from("<H", cd_data, pos+30)[0]
        comment_len   = struct.unpack_from("<H", cd_data, pos+32)[0]
        lh_offset_32  = struct.unpack_from("<I", cd_data, pos+42)[0]
        fname = cd_data[pos+46:pos+46+fname_len].decode("utf-8", errors="replace")
        extra = cd_data[pos+46+fname_len:pos+46+fname_len+extra_len]
        comp_size, _, lh_offset = _resolve_zip64_extra(
            extra, comp_size_32, uncomp_size_32, lh_offset_32)
        if fname == target or fname.endswith("/"+target) or fname.endswith("\\"+target):
            return {"fname": fname, "compression": comp_method,
                    "compressed_size": comp_size, "local_header_offset": lh_offset}
        pos += 46 + fname_len + extra_len + comment_len
    return None


def _list_cd(cd_data: bytes, n: int = 50) -> None:
    pos = 0
    count = 0
    while pos + 46 <= len(cd_data) and count < n:
        if cd_data[pos:pos+4] != CD_ENTRY_SIG: break
        fname_len   = struct.unpack_from("<H", cd_data, pos+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, pos+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos+32)[0]
        comp_size   = struct.unpack_from("<I", cd_data, pos+20)[0]
        fname = cd_data[pos+46:pos+46+fname_len].decode("utf-8", errors="replace")
        print(f"    {fname}  ({comp_size/1e6:.1f} MB compressed)")
        pos += 46 + fname_len + extra_len + comment_len
        count += 1


def _local_data_offset(ftp_path: str, lh_abs: int,
                       fallback_fname: str = "") -> int:
    """
    Read local file header to find where compressed data begins.
    Reads 512 bytes (covers header 30b + long filename + extra field in one shot).
    """
    print(f"  Reading local header at offset {lh_abs:,} ...")
    lh = ftp_read_range(ftp_path, lh_abs, 512)
    if len(lh) < 30:
        if fallback_fname:
            print(f"  [!] Local header read returned {len(lh)} bytes -- using CD fallback")
            return _local_data_offset_fallback(lh_abs, fallback_fname)
        raise ValueError(
            f"Local header: got only {len(lh)} bytes at offset {lh_abs:,}.\n"
            f"  Raw: {lh!r}\n"
            f"  FTP REST may not work at this offset."
        )
    if lh[:4] != LOCAL_ENTRY_SIG:
        raise ValueError(
            f"Local header sig mismatch at {lh_abs:,}: got {lh[:4]!r}, "
            f"expected {LOCAL_ENTRY_SIG!r}"
        )
    fname_len = struct.unpack_from("<H", lh, 26)[0]
    extra_len = struct.unpack_from("<H", lh, 28)[0]
    data_start = lh_abs + 30 + fname_len + extra_len
    print(f"  Local header OK: fname_len={fname_len}, extra_len={extra_len}, "
          f"data_start={data_start:,}")
    return data_start


def _local_data_offset_fallback(lh_abs: int, fname: str) -> int:
    """
    If we cannot read the local header (FTP REST failure at large offset),
    estimate data_start using the filename length from the central directory.
    Extra field length is guessed as 20 (typical ZIP64 local extra field).
    This is an approximation -- if it fails, the decompressed output will be garbage.
    """
    fname_len = len(fname.encode("utf-8"))
    extra_len_guess = 20   # typical ZIP64 extended info extra field in local header
    data_start = lh_abs + 30 + fname_len + extra_len_guess
    print(f"  [fallback] Estimating data_start={data_start:,} "
          f"(fname_len={fname_len}, extra_len={extra_len_guess} assumed)")
    return data_start


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(ftp_path: str, target: str, output_path: Path) -> bool:
    print(f"\n  FTP: {ftp_path}")
    print(f"  Target: {target}")

    print("  [1/4] Getting ZIP size ...")
    file_size = ftp_file_size(ftp_path)
    print(f"  ZIP size: {file_size/1e9:.2f} GB")

    tail_len   = min(65536, file_size)
    tail_start = file_size - tail_len
    print(f"  [2/4] Reading last {tail_len//1024}KB (EOCD search) ...")
    tail = ftp_read_range(ftp_path, tail_start, tail_len)

    eocd_pos = _find_eocd(tail)
    cd_offset, cd_size = _parse_eocd(tail, eocd_pos)

    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        print("  ZIP64 format -- reading ZIP64 EOCD ...")
        cd_offset, cd_size = _parse_zip64_eocd(ftp_path, tail, file_size)

    print(f"  Central dir: offset={cd_offset:,}, size={cd_size/1e6:.1f} MB")

    print(f"  [3/4] Reading central directory ({cd_size/1e6:.1f} MB) ...")
    cd_data = ftp_read_range(ftp_path, cd_offset, cd_size)

    entry = _search_cd(cd_data, target)
    if entry is None:
        print(f"  '{target}' not found. First 50 files in ZIP:")
        _list_cd(cd_data)
        return False

    print(f"  Found: {entry['fname']}")
    comp_size = entry["compressed_size"]
    print(f"  Compressed: {comp_size/1e6:.1f} MB, "
          f"method={'deflate' if entry['compression']==8 else entry['compression']}")

    data_start = _local_data_offset(
        ftp_path, entry["local_header_offset"], fallback_fname=entry["fname"]
    )

    print(f"  [4/4] Downloading compressed data ({comp_size/1e6:.1f} MB) ...")
    compressed = ftp_read_range(ftp_path, data_start, comp_size, show_progress=True)

    print("  Decompressing ...")
    if entry["compression"] == 0:
        raw = compressed
    elif entry["compression"] == 8:
        raw = zlib.decompress(compressed, wbits=-15)
    else:
        raise ValueError(f"Unsupported compression method: {entry['compression']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    print(f"  Saved: {output_path}  ({len(raw)/1e6:.1f} MB)")
    return True


def main():
    accession = sys.argv[1] if len(sys.argv) > 1 else "PXD014791"
    if accession not in FTP_PATHS:
        print(f"Unknown accession: {accession}. Known: {list(FTP_PATHS)}")
        sys.exit(1)

    ftp_path  = FTP_PATHS[accession]
    out_path  = OUTPUT_DIR / accession / TARGET_FILE

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    try:
        ok = extract(ftp_path, TARGET_FILE, out_path)
        sys.exit(0 if ok else 1)
    except Exception as e:
        import traceback
        print(f"\nError: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
