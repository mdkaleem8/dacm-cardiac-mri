"""Download the public CMRxMotion training and validation zips from Synapse.

Synapse needs a (free) account for downloads. Put a Personal Access Token in
the environment variable SYNAPSE_AUTH_TOKEN, or in a one-line text file passed
with --token_file. The token is only sent to Synapse.

  python scripts/download_cmrxmotion.py --out ../data/CMRxMotion --token_file ../data/synapse_token.txt
"""
import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://repo-prod.prod.sagebase.org/repo/v1"
FILE = "https://file-prod.prod.sagebase.org/file/v1"
FILES = {"syn31469794": "CMRxMotion_Training.zip", "syn32414018": "CMRxMotion_Validation.zip"}


def call(url, token, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.replace(dest)


def safe_extract(zp, out):
    root = out.resolve()
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            target = (out / m.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
        z.extractall(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--token_file", default="")
    a = ap.parse_args()
    token = os.environ.get("SYNAPSE_AUTH_TOKEN", "")
    if not token and a.token_file and Path(a.token_file).exists():
        token = Path(a.token_file).read_text().strip()
    if not token:
        sys.exit("No Synapse token: set SYNAPSE_AUTH_TOKEN or pass --token_file")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for syn, name in FILES.items():
        dest = out / name
        if not dest.exists():
            ent = call(f"{REPO}/entity/{syn}", token)
            res = call(f"{FILE}/fileHandle/batch", token, {
                "requestedFiles": [{"fileHandleId": ent["dataFileHandleId"], "associateObjectId": syn,
                                    "associateObjectType": "FileEntity"}],
                "includePreSignedURLs": True, "includeFileHandles": False})["requestedFiles"][0]
            if res.get("failureCode"):
                sys.exit(f"{syn}: Synapse refused the download ({res['failureCode']}). "
                         "Check that the token has 'download' scope.")
            print(f"Downloading {name}")
            download(res["preSignedURL"], dest)
        print(f"Extracting {name}")
        safe_extract(dest, out / dest.stem)
    print("CMRxMotion ready in", out)


if __name__ == "__main__":
    main()
