#!/usr/bin/env python3
"""
Upload File ke Synology NAS (DMS)
==================================
Upload satu file lokal ke folder tertentu di NAS via Synology File Station API.
Konfigurasi diambil dari .env (DMS_BASE_URL, DMS_USERNAME, DMS_PASSWORD, NAS_ROOT).

Usage:
    python upload_to_nas.py <local_file> --folder "<subfolder relatif NAS_ROOT>"
    python upload_to_nas.py "../excel_resources/Data Dashboard KESGI.xlsx" --folder "Report KESGI"
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_env(env_path: str = None) -> dict:
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
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


class SynologyDMSUploader:
    """Client Synology File Station API untuk create folder + upload file."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
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
                print(f"✅ Login berhasil (SID: {self.sid[:8]}...)")
                return True
            print(f"❌ Login gagal: {data}")
            return False
        except Exception as e:
            print(f"❌ Login error: {e}")
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

    def folder_exists(self, folder_path: str) -> bool:
        """Cek apakah folder sudah ada via List API."""
        r = self.session.get(
            f"{self.base_url}/webapi/entry.cgi",
            params={
                "api": "SYNO.FileStation.List",
                "version": "2",
                "method": "list",
                "folder_path": folder_path,
            },
            timeout=30,
        )
        data = r.json()
        return bool(data.get("success"))

    def create_folder(self, parent_path: str, name: str) -> bool:
        """Buat folder baru. Idempotent: no-op kalau sudah ada."""
        target = f"{parent_path.rstrip('/')}/{name}"
        if self.folder_exists(target):
            print(f"📁 Folder sudah ada: {target}")
            return True

        r = self.session.get(
            f"{self.base_url}/webapi/entry.cgi",
            params={
                "api": "SYNO.FileStation.CreateFolder",
                "version": "2",
                "method": "create",
                "folder_path": parent_path,
                "name": name,
            },
            timeout=30,
        )
        data = r.json()
        if data.get("success"):
            print(f"✅ Folder dibuat: {target}")
            return True
        print(f"❌ Gagal buat folder {target}: {data}")
        return False

    def upload_file(self, local_path: Path, dest_folder: str, overwrite: bool = True) -> bool:
        """Upload satu file ke folder tujuan di NAS."""
        if not local_path.exists():
            print(f"❌ File lokal tidak ditemukan: {local_path}")
            return False

        print(f"⬆️  Uploading {local_path.name} ({local_path.stat().st_size / 1024:.0f} KB) → {dest_folder} ...")

        with open(local_path, "rb") as f:
            files = {"file": (local_path.name, f, "application/octet-stream")}
            data = {
                "api": "SYNO.FileStation.Upload",
                "version": "2",
                "method": "upload",
                "path": dest_folder,
                "create_parents": "true",
                "overwrite": "true" if overwrite else "false",
                "_sid": self.sid,
            }
            r = self.session.post(
                f"{self.base_url}/webapi/entry.cgi",
                data=data,
                files=files,
                timeout=300,
            )

        try:
            result = r.json()
        except ValueError:
            print(f"❌ Upload gagal, response bukan JSON: {r.text[:300]}")
            return False

        if result.get("success"):
            print(f"✅ Upload sukses: {dest_folder}/{local_path.name}")
            return True
        print(f"❌ Upload gagal: {result}")
        return False

    def file_exists(self, folder_path: str, filename: str) -> bool:
        """Cek apakah file sudah ada di folder NAS (verifikasi pasca-upload)."""
        r = self.session.get(
            f"{self.base_url}/webapi/entry.cgi",
            params={
                "api": "SYNO.FileStation.List",
                "version": "2",
                "method": "list",
                "folder_path": folder_path,
            },
            timeout=30,
        )
        data = r.json()
        if not data.get("success"):
            return False
        names = [f["name"] for f in data.get("data", {}).get("files", [])]
        return filename in names


def main():
    parser = argparse.ArgumentParser(description="Upload file lokal ke Synology NAS")
    parser.add_argument("local_file", type=str, help="Path file lokal yang akan diupload")
    parser.add_argument("--folder", type=str, required=True,
                         help="Subfolder tujuan relatif terhadap NAS_ROOT (dibuat otomatis jika belum ada)")
    args = parser.parse_args()

    local_path = Path(args.local_file).resolve()
    if not local_path.exists():
        print(f"❌ File tidak ditemukan: {local_path}")
        sys.exit(1)

    config = load_env()
    base_url = config.get("DMS_BASE_URL")
    username = config.get("DMS_USERNAME")
    password = config.get("DMS_PASSWORD")
    nas_root = config.get("NAS_ROOT")

    if not all([base_url, username, password, nas_root]):
        print("❌ .env tidak lengkap (perlu DMS_BASE_URL, DMS_USERNAME, DMS_PASSWORD, NAS_ROOT)")
        sys.exit(1)

    client = SynologyDMSUploader(base_url, username, password)
    if not client.login():
        sys.exit(1)

    try:
        dest_folder = f"{nas_root.rstrip('/')}/{args.folder}"

        if not client.create_folder(nas_root, args.folder):
            sys.exit(1)

        if not client.upload_file(local_path, dest_folder):
            sys.exit(1)

        print("🔍 Verifikasi file di NAS...")
        if client.file_exists(dest_folder, local_path.name):
            print(f"✅ Terverifikasi: {dest_folder}/{local_path.name}")
        else:
            print(f"⚠️  Upload dilaporkan sukses tapi file tidak ditemukan saat verifikasi di {dest_folder}")
            sys.exit(1)
    finally:
        client.logout()


if __name__ == "__main__":
    main()
