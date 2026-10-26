"""
DMS NAS Explorer (Synology DSM)
Membaca semua folder dan file dari Synology NAS via File Station API.
Konfigurasi diambil dari .env
"""

import sys
import json
import urllib3
import requests
from pathlib import Path
from typing import Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Load .env ────────────────────────────────────────────────────────────────
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


# ── Synology File Station Client ─────────────────────────────────────────────
class SynologyDMSExplorer:
    """Explorer untuk Synology NAS via File Station API."""

    def __init__(self, base_url: str, username: str, password: str, root_path: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.root_path = root_path
        self.session = requests.Session()
        self.session.verify = False
        self.sid: Optional[str] = None

    # ── Auth ─────────────────────────────────────────────────────────────────
    def login(self) -> bool:
        """Login ke Synology DSM."""
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
            else:
                print(f"❌ Login gagal: {data}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    def logout(self):
        """Logout dari Synology DSM."""
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

    # ── File Station API ─────────────────────────────────────────────────────
    def _api_call(self, method: str, extra_params: dict = None) -> dict:
        """Panggil Synology File Station API."""
        params = {
            "api": "SYNO.FileStation.List",
            "version": "2",
            "method": method,
        }
        if extra_params:
            params.update(extra_params)

        r = self.session.get(
            f"{self.base_url}/webapi/entry.cgi",
            params=params,
            timeout=30,
        )
        return r.json()

    def list_shares(self) -> list[dict]:
        """List semua shared folders."""
        data = self._api_call("list_share")
        if data.get("success"):
            return data["data"]["shares"]
        print(f"  ⚠️ list_share gagal: {data}")
        return []

    def list_folder(self, folder_path: str) -> list[dict]:
        """List isi satu folder."""
        data = self._api_call("list", {"folder_path": folder_path})
        if data.get("success"):
            return data["data"]["files"]
        else:
            error = data.get("error", {})
            if error.get("code") not in (408, 900):
                print(f"  ⚠️ list {folder_path}: {error}")
            return []

    # ── Recursive listing ────────────────────────────────────────────────────
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
        print(f"{indent}📂 {folder_path or '/'}")

        items = self.list_folder(folder_path)
        dirs = [i for i in items if i.get("isdir")]
        files = [i for i in items if not i.get("isdir")]

        for f in files:
            size = f.get("additional", {}).get("size", 0)
            entry = {
                "name": f["name"],
                "path": f["path"],
                "is_dir": False,
                "size": size,
                "modified": f.get("additional", {}).get("time", {}).get("last_modified", ""),
            }
            results.append(entry)
            print(f"{indent}  📄 {f['name']} ({size:,} bytes)")

        for d in dirs:
            entry = {
                "name": d["name"],
                "path": d["path"],
                "is_dir": True,
                "size": 0,
                "modified": d.get("additional", {}).get("time", {}).get("last_modified", ""),
            }
            results.append(entry)
            print(f"{indent}  📁 {d['name']}/")

            if depth < max_depth:
                self.list_all_recursive(d["path"], depth + 1, max_depth, results)

        return results

    # ── Download file ────────────────────────────────────────────────────────
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
            if r.status_code == 200 and "application/json" not in r.headers.get("content-type", ""):
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False
        except Exception as e:
            print(f"  ❌ Download error: {e}")
            return False


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    config = load_env()

    base_url = config.get("DMS_BASE_URL", "")
    username = config.get("DMS_USERNAME", "")
    password = config.get("DMS_PASSWORD", "")
    root_path = config.get("NAS_ROOT", "")

    if not all([base_url, username, password, root_path]):
        print("❌ .env tidak lengkap!")
        sys.exit(1)

    explorer = SynologyDMSExplorer(base_url, username, password, root_path)

    # ── Login ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🔐 LOGIN")
    print("=" * 60)
    if not explorer.login():
        sys.exit(1)

    # ── List shares ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📦 SHARED FOLDERS")
    print("=" * 60)
    shares = explorer.list_shares()
    for s in shares:
        print(f"  📁 {s['name']}")

    # ── List root folder ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"📂 SCANNING: {root_path}")
    print("=" * 60)

    all_items = explorer.list_all_recursive(root_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    dirs = [i for i in all_items if i["is_dir"]]
    files = [i for i in all_items if not i["is_dir"]]
    total_size = sum(f["size"] for f in files)

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Total folder : {len(dirs)}")
    print(f"   Total file   : {len(files)}")
    print(f"   Total size   : {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

    # ── Save JSON ────────────────────────────────────────────────────────────
    output_file = Path(__file__).parent / "dms_structure.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Struktur disimpan ke: {output_file}")

    # ── Tree level 1 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📁 FOLDER TREE (level 1)")
    print("=" * 60)
    root_depth = root_path.rstrip("/").count("/")
    for d in dirs:
        depth = d["path"].count("/") - root_depth
        if depth <= 1:
            print(f"  {'  ' * depth}📁 {d['name']}/")

    print("\n📄 FILE DI ROOT:")
    for f in files:
        depth = f["path"].count("/") - root_depth
        if depth <= 1:
            print(f"  📄 {f['name']} ({f['size']:,} bytes)")

    # ── Logout ───────────────────────────────────────────────────────────────
    explorer.logout()
    print("\n👋 Selesai.")


if __name__ == "__main__":
    main()
