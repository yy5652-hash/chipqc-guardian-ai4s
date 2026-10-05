"""Download the Organ-on-a-Chip Image Dataset from Zenodo, verify it and unpack it into data/raw/.

    python scripts/download_data.py        # 6.7 GB; resumes an interrupted download

Source: Movčana et al., doi:10.5281/zenodo.10203721, CC BY 4.0.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import ARCHIVE_MD5, ZENODO_RECORD  # noqa: E402

RAW = ROOT / "data/raw"
BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"


def fetch(name: str) -> Path:
    target = RAW / name
    done = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(f"{BASE}/{name}/content", headers={"Range": f"bytes={done}-"} if done else {})
    try:
        with urllib.request.urlopen(request) as response, open(target, "ab" if done else "wb") as out:
            total = done + int(response.headers.get("Content-Length", 0))
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                print(f"\r{name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
        print()
    except urllib.error.HTTPError as e:
        if e.code != 416:                       # 416: the file is already complete
            raise
    return target


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    fetch("OOC_datasheet.xlsx")
    archive = fetch("OOC_image_dataset.zip")
    md5 = hashlib.md5()
    with open(archive, "rb") as f:
        while chunk := f.read(1 << 22):
            md5.update(chunk)
    if md5.hexdigest() != ARCHIVE_MD5:
        sys.exit(f"checksum mismatch: {md5.hexdigest()} (expected {ARCHIVE_MD5}); delete {archive} and run again")
    print("checksum verified; unpacking (the system `unzip` cannot read this ZIP64 archive, Python can)")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW)
    print("images in", RAW / "OOC_image_dataset")
