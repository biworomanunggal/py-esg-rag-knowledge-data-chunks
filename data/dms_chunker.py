#!/usr/bin/env python3
"""
DMS NAS → Chunker Pipeline
============================
Download PDF dari Synology NAS, chunk, dan siap untuk insert ke Qdrant.

Pipeline:
1. Connect ke DMS NAS (Synology File Station API)
2. List semua folder & file di root path
3. Download PDF files
4. Chunk setiap PDF (menggunakan pdf_chunker yang sudah ada)
5. Simpan hasil chunk ke JSON

Usage:
    python dms_chunker.py                          # Process semua sektor
    python dms_chunker.py --sector "Perbankan"     # Process sektor tertentu
    python dms_chunker.py --list-sectors           # List semua sektor
    python dms_chunker.py --download-only          # Download tanpa chunk
    python dms_chunker.py --force                  # Force re-chunk
"""

import os
import sys
import json
import re
import time
import hashlib
import requests
import urllib3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add parent paths untuk import pdf_chunker
SCRIPT_DIR = Path(__file__).parent
CHUNKER_DIR = SCRIPT_DIR.parent / "chunks"
sys.path.insert(0, str(CHUNKER_DIR))

# ── Load .env ────────────────────────────────────────────────────────────────
def load_env(env_path: str = None) -> dict:
    if env_path is None:
        env_path = SCRIPT_DIR / ".env"
    config = {}
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            config[key] = value
    return config


# ── Synology DMS Client ─────────────────────────────────────────────────────
class SynologyDMS:
    """Synology NAS client via File Station API."""

    def __init__(self, base_url: str, username: str, password: str, root_path: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.root_path = root_path
        self.session = requests.Session()
        self.session.verify = False
        self.sid: Optional[str] = None

    def login(self) -> bool:
        try:
            r = self.session.get(
                f"{self.base_url}/webapi/auth.cgi",
                params={
                    "api": "SYNO.API.Auth",
                    "version": "6",
                    "method": "login",
                    "account": self.username,
                    "passwd": self.password,
                    "session": "FileStation",
                    "format": "cookie",
                },
                timeout=15,
            )
            data = r.json()
            if data.get("success"):
                self.sid = data["data"]["sid"]
                print(f"✅ DMS Login berhasil (SID: {self.sid[:8]}...)")
                return True
            print(f"❌ DMS Login gagal: {data}")
            return False
        except Exception as e:
            print(f"❌ DMS Login error: {e}")
            return False

    def logout(self):
        if self.sid:
            try:
                self.session.get(
                    f"{self.base_url}/webapi/auth.cgi",
                    params={
                        "api": "SYNO.API.Auth",
                        "version": "6",
                        "method": "logout",
                        "session": "FileStation",
                    },
                    timeout=10,
                )
            except Exception:
                pass
            self.sid = None

    def list_folder(self, folder_path: str) -> List[dict]:
        """List isi satu folder."""
        params = {
            "api": "SYNO.FileStation.List",
            "version": "2",
            "method": "list",
            "folder_path": folder_path,
        }
        r = self.session.get(
            f"{self.base_url}/webapi/entry.cgi",
            params=params,
            timeout=30,
        )
        data = r.json()
        if data.get("success"):
            return data["data"]["files"]
        return []

    def list_all_recursive(
        self,
        folder_path: str = "",
        depth: int = 0,
        max_depth: int = 10,
        results: list = None,
    ) -> list[dict]:
        """List semua folder & file secara rekursif."""
        if results is None:
            results = []
        indent = "  " * depth

        items = self.list_folder(folder_path)
        dirs = [i for i in items if i.get("isdir")]
        files = [i for i in items if not i.get("isdir")]

        for f in files:
            entry = {
                "name": f["name"],
                "path": f["path"],
                "is_dir": False,
                "size": f.get("additional", {}).get("size", 0),
            }
            results.append(entry)
            ext = Path(f["name"]).suffix.lower()
            print(f"{indent}  📄 {f['name']}")

        for d in dirs:
            entry = {
                "name": d["name"],
                "path": d["path"],
                "is_dir": True,
                "size": 0,
            }
            results.append(entry)
            print(f"{indent}  📁 {d['name']}/")

            if depth < max_depth:
                self.list_all_recursive(d["path"], depth + 1, max_depth, results)

        return results

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download satu file dari NAS."""
        try:
            params = {
                "api": "SYNO.FileStation.Download",
                "version": "2",
                "method": "download",
                "path": remote_path,
                "mode": "download",
            }
            r = self.session.get(
                f"{self.base_url}/webapi/entry.cgi",
                params=params,
                stream=True,
                timeout=300,
            )
            content_type = r.headers.get("content-type", "")
            if r.status_code == 200 and "application/json" not in content_type:
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"  ❌ Download failed: HTTP {r.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Download error: {e}")
            return False


# ── Sector Mapping (DMS → Qdrant) ───────────────────────────────────────────
# DMS menggunakan nama sektor dalam Bahasa Indonesia,
# Qdrant menggunakan nama sektor dalam Bahasa Inggris
SECTOR_MAP_DMS_TO_QDRANT = {
    "Bahan Baku": "Bahan Baku",
    "E-Commerce": "E-Commerce",
    "Energi (Migas)": "Energy",
    "FMCG": "FMCG",
    "Garmen & Tekstil": "Garmen & Tekstil",
    "Infrastruktur-Konstruksi Berat": "Infrastruktur-Konstruksi Berat",
    "Keuangan Inklusif": "Keuangan Inklusif",
    "Konglomerasi": "Konglomerasi",
    "Listrik & Utilitas": "Listrik & Utilitas",
    "Perbankan": "Finance",
    "Perkebunan": "Plantation",
    "Pertambangan": "Mining",
    "Telekomunikasi": "Telecomunication",
    "Transportasi dan Logistik": "Transportation",
}


# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="DMS NAS → Chunker Pipeline")
    parser.add_argument("--sector", type=str, help="Sector tertentu (comma-separated)")
    parser.add_argument("--list-sectors", action="store_true", help="List semua sektor di DMS")
    parser.add_argument("--download-only", action="store_true", help="Download tanpa chunk")
    parser.add_argument("--force", action="store_true", help="Force re-chunk")
    parser.add_argument("--max-files", type=int, help="Limit jumlah file (untuk testing)")
    args = parser.parse_args()

    # Load config
    config = load_env()
    base_url = config.get("DMS_BASE_URL", "")
    username = config.get("DMS_USERNAME", "")
    password = config.get("DMS_PASSWORD", "")
    root_path = config.get("NAS_ROOT", "")

    if not all([base_url, username, password, root_path]):
        print("❌ .env tidak lengkap!")
        sys.exit(1)

    # Output directories
    download_dir = SCRIPT_DIR / "downloads"
    chunk_output_dir = SCRIPT_DIR.parent / "chunked_data"

    # Connect ke DMS
    print("=" * 60)
    print("🔐 DMS NAS CONNECT")
    print("=" * 60)
    dms = SynologyDMS(base_url, username, password, root_path)
    if not dms.login():
        sys.exit(1)

    # List root folder
    print(f"\n📂 Scanning: {root_path}")
    print("=" * 60)
    all_items = dms.list_all_recursive(root_path)

    # Group by sector (level 1 folder)
    sectors = {}
    for item in all_items:
        if item["is_dir"]:
            continue
        # Extract sector from path: /DEVELOPMENT/KESGIDATA/KESGI Sync data/{sector}/...
        rel_path = item["path"].replace(root_path, "").lstrip("/")
        parts = rel_path.split("/")
        if len(parts) < 1:
            continue
        sector_name = parts[0]
        if sector_name not in sectors:
            sectors[sector_name] = []
        sectors[sector_name].append(item)

    # List sectors
    if args.list_sectors:
        print("\n" + "=" * 60)
        print("📁 SECTORS di DMS")
        print("=" * 60)
        for sector, files in sorted(sectors.items()):
            qdrant_sector = SECTOR_MAP_DMS_TO_QDRANT.get(sector, sector)
            print(f"  {sector} → {qdrant_sector} ({len(files)} files)")
        print(f"\nTotal: {sum(len(f) for f in sectors.values())} files di {len(sectors)} sektor")
        dms.logout()
        return

    # Filter sectors
    selected_sectors = None
    if args.sector:
        selected_sectors = [s.strip() for s in args.sector.split(",")]
        # Map input ke DMS sector names (case-insensitive)
        all_sector_names = list(sectors.keys())
        selected_sectors = [
            s for s in all_sector_names
            if any(s.lower() == sel.lower() or sel.lower() in s.lower() for sel in selected_sectors)
        ]
        print(f"\nFilter: {selected_sectors}")

    # Process
    print("\n" + "=" * 60)
    print("📦 PROCESSING")
    print("=" * 60)

    total_downloaded = 0
    total_chunked = 0
    total_skipped = 0
    file_count = 0

    for sector_name, files in sorted(sectors.items()):
        if selected_sectors and sector_name not in selected_sectors:
            continue

        qdrant_sector = SECTOR_MAP_DMS_TO_QDRANT.get(sector_name, sector_name)
        print(f"\n{'─' * 60}")
        print(f"📁 {sector_name} → {qdrant_sector} ({len(files)} files)")
        print(f"{'─' * 60}")

        for file_info in files:
            if args.max_files and file_count >= args.max_files:
                print(f"\n⚠️ Max files limit reached ({args.max_files})")
                break

            file_count += 1
            filename = file_info["name"]
            remote_path = file_info["path"]

            # Skip non-PDF
            ext = Path(filename).suffix.lower()
            if ext not in (".pdf", ".docx"):
                continue

            # Local paths
            rel_path = remote_path.replace(root_path, "").lstrip("/")
            local_path = download_dir / rel_path

            # Qdrant sector output path
            safe_name = re.sub(r'[^\w\-_]', '_', Path(filename).stem.lower())
            chunk_output_path = chunk_output_dir / qdrant_sector / f"{safe_name}_chunks.json"

            # Skip if already chunked (unless --force)
            if not args.force and not args.download_only and chunk_output_path.exists():
                print(f"  ⏭️  Skip (already chunked): {filename}")
                total_skipped += 1
                continue

            # Download
            if not local_path.exists() or args.force:
                print(f"  ⬇️  Downloading: {filename}")
                if not dms.download_file(remote_path, str(local_path)):
                    print(f"  ❌ Download failed, skipping")
                    continue
                total_downloaded += 1
            else:
                print(f"  ✅ Already downloaded: {filename}")

            if args.download_only:
                continue

            # Chunk
            if ext == ".pdf":
                print(f"  📄 Chunking: {filename}")
                try:
                    from pdf_chunker import PDFChunker
                    chunker = PDFChunker(str(local_path))
                    # Override sector dengan Qdrant sector name
                    chunker.sector = qdrant_sector
                    chunks = chunker.process()
                    chunker.save(chunk_output_path)
                    total_chunked += 1
                    print(f"  ✅ {len(chunks)} chunks → {chunk_output_path}")
                except Exception as e:
                    print(f"  ❌ Chunk error: {e}")
            elif ext == ".docx":
                print(f"  📄 DOCX chunking: {filename}")
                # TODO: Implement DOCX chunking
                print(f"  ⚠️ DOCX chunking not yet implemented, skipping")

        if args.max_files and file_count >= args.max_files:
            break

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Downloaded : {total_downloaded}")
    print(f"   Chunked    : {total_chunked}")
    print(f"   Skipped    : {total_skipped}")

    dms.logout()
    print("\n👋 Selesai.")


if __name__ == "__main__":
    main()
