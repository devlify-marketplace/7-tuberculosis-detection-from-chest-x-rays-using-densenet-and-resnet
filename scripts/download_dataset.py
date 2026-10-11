#!/usr/bin/env python3
"""Download the official NLM Shenzhen and Montgomery CXR PNG files.

Dataset source: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
The original radiographs are several gigabytes in total and are intentionally not bundled.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from PIL import Image

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
DATASETS = {
    "shenzhen": BASE + "Shenzhen-Hospital-CXR-Set/CXR_png/index.html",
    "montgomery": BASE + "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/index.html",
}

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def list_pngs(index_url: str) -> list[str]:
    req = urllib.request.Request(index_url, headers={"User-Agent": "tb-cxr-research/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        html = response.read().decode("utf-8", "replace")
    parser = LinkParser(); parser.feed(html)
    urls = []
    for link in parser.links:
        if urlparse(link).path.lower().endswith(".png"):
            urls.append(urljoin(index_url, link))
    if not urls:
        raise RuntimeError(f"No PNG files found in official listing {index_url}")
    return sorted(set(urls))


def download_one(item: tuple[str, Path]) -> tuple[str, str, int]:
    url, target = item
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        try:
            with Image.open(target) as img: img.verify()
            return target.name, "already-valid", target.stat().st_size
        except Exception:
            target.unlink(missing_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tb-cxr-research/1.0"})
            digest = hashlib.sha256(); total = 0
            with urllib.request.urlopen(req, timeout=120) as response, temp.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block: break
                    out.write(block); digest.update(block); total += len(block)
            if total == 0:
                raise IOError("empty response")
            with Image.open(temp) as img: img.verify()
            temp.replace(target)
            return target.name, digest.hexdigest(), total
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt < 2: time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", choices=["all", *DATASETS], default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    root = args.root.resolve(); selected = DATASETS if args.source == "all" else {args.source: DATASETS[args.source]}
    manifest = []
    for source, index_url in selected.items():
        urls = list_pngs(index_url)
        print(f"{source}: {len(urls)} images listed at the official NLM index", flush=True)
        # Subdirectory names match the official folder structure expected by the provenance note.
        destination = root / "data" / "raw" / source / "CXR_png"
        items = [(url, destination / Path(urlparse(url).path).name) for url in urls]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_one, item) for item in items]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                name, digest, size = future.result()
                manifest.append((source, name, digest, size))
                if count % 25 == 0 or count == len(futures):
                    print(f"  {source}: {count}/{len(futures)} complete", flush=True)
    checksum_path = root / "data" / "raw" / "SHA256SUMS.txt"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    with checksum_path.open("w", encoding="utf-8") as out:
        out.write("# SHA-256 checksums for downloaded NLM PNG files\n")
        for source, name, digest, size in sorted(manifest):
            if digest == "already-valid":
                path = root / "data" / "raw" / source / "CXR_png" / name
                hasher = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""): hasher.update(block)
                digest = hasher.hexdigest()
            rel = Path("data/raw") / source / "CXR_png" / name
            out.write(f"{digest}  {rel.as_posix()}  # {size} bytes\n")
    total_bytes = sum(row[3] for row in manifest)
    print(f"Downloaded/verified {len(manifest)} images ({total_bytes / (1024**3):.2f} GiB). Checksums: {checksum_path}")

if __name__ == "__main__":
    main()
