"""
Run this FIRST to diagnose what's reachable from your WSL before attempting download.
Usage: pipenv run python diagnose.py
"""

import sys
import socket
import urllib.request
import json
from ftplib import FTP

TESTS = []

def test(label):
    def decorator(fn):
        TESTS.append((label, fn))
        return fn
    return decorator


@test("DNS resolution for ftp.pride.ebi.ac.uk")
def _():
    ip = socket.gethostbyname("ftp.pride.ebi.ac.uk")
    return f"OK -> {ip}"


@test("DNS resolution for www.ebi.ac.uk")
def _():
    ip = socket.gethostbyname("www.ebi.ac.uk")
    return f"OK -> {ip}"


@test("FTP connection to ftp.pride.ebi.ac.uk (port 21)")
def _():
    ftp = FTP("ftp.pride.ebi.ac.uk", timeout=15)
    ftp.login()
    welcome = ftp.getwelcome()[:60]
    ftp.quit()
    return f"OK -> {welcome}"


@test("FTP: list /pride/data/archive/2022/01/PXD014791/")
def _():
    ftp = FTP("ftp.pride.ebi.ac.uk", timeout=20)
    ftp.login()
    ftp.set_pasv(True)
    entries = ftp.nlst("/pride/data/archive/2022/01/PXD014791/")
    ftp.quit()
    return f"OK -> {len(entries)} entries: {entries[:5]}"


@test("HTTPS GET pride REST API v2")
def _():
    url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD014791"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        status = resp.status
        body = resp.read(200)
    return f"HTTP {status}, body[:200]: {body}"


@test("HTTPS GET proteomecentral (ProteomeXchange) API")
def _():
    url = "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD014791&outputMode=json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        status = resp.status
        body = json.loads(resp.read())
    return f"HTTP {status}, keys: {list(body.keys())[:6]}"


if __name__ == "__main__":
    print("=" * 60)
    print("Network diagnostic for PRIDE download")
    print("=" * 60)
    for label, fn in TESTS:
        print(f"\n[TEST] {label}")
        try:
            result = fn()
            print(f"  PASS: {result}")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print("Share the output above to debug download issues.")
