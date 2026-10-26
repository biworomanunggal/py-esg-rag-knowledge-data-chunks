#!/usr/bin/env python3
"""
ESG Data Pipeline Orchestrator
===============================
Satu entry point untuk seluruh pipeline data: dari dokumen di NAS sampai
collection Qdrant siap dipakai chatbot.

Menjalankan berurutan (tiap step idempotent, aman di-rerun):
    1. data/dms_chunker.py        → download PDF dari NAS + chunking
    2. qdrant_insert/insertdata.py        → insert chunk PDF ke Qdrant (esg_reports)
    3. qdrant_insert/insertdata_excel.py  → insert chunk Excel ke Qdrant (esg_data_reports)
    4. py-esg-rag-chatbot-api/sync_companies.py → rebuild collection esg_companies

Usage:
    python run_pipeline.py                      # full pipeline, semua sektor
    python run_pipeline.py --sector Perbankan    # hanya 1 sektor (step 1 & 2)
    python run_pipeline.py --skip-nas            # mulai dari step 2 (chunk sudah ada)
    python run_pipeline.py --skip-excel          # lewati step 3
    python run_pipeline.py --only sync           # jalankan 1 step saja: nas|insert|excel|sync
    python run_pipeline.py --dry-run             # tampilkan rencana tanpa eksekusi
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY_ESG_RAG_DIR = Path(__file__).parent
DATA_DIR = PY_ESG_RAG_DIR / "data"
QDRANT_INSERT_DIR = PY_ESG_RAG_DIR / "qdrant_insert"
CHATBOT_API_DIR = PY_ESG_RAG_DIR.parent / "py-esg-rag-chatbot-api"

# Tiap project punya venv sendiri — pakai python dari venv masing-masing
PY_ESG_RAG_PYTHON = PY_ESG_RAG_DIR / ".venv" / "bin" / "python3"
CHATBOT_API_PYTHON = CHATBOT_API_DIR / ".venv" / "bin" / "python3"

# dms_chunker.py (step 1) pakai nama sektor NAS (Bahasa Indonesia).
# insertdata.py (step 2) pakai nama folder di chunked_data/, yaitu nama Qdrant (Bahasa Inggris).
# Harus sinkron dengan SECTOR_MAP_DMS_TO_QDRANT di data/dms_chunker.py.
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
_QDRANT_TO_DMS = {v: k for k, v in SECTOR_MAP_DMS_TO_QDRANT.items()}


def resolve_sector_names(sector: str) -> tuple[str, str]:
    """Terima nama sektor NAS atau Qdrant, kembalikan (nama_nas, nama_qdrant)."""
    if sector in SECTOR_MAP_DMS_TO_QDRANT:
        return sector, SECTOR_MAP_DMS_TO_QDRANT[sector]
    if sector in _QDRANT_TO_DMS:
        return _QDRANT_TO_DMS[sector], sector
    valid = sorted(set(SECTOR_MAP_DMS_TO_QDRANT) | set(_QDRANT_TO_DMS))
    print(f"❌ Sektor '{sector}' tidak dikenal. Pilihan valid:\n  " + "\n  ".join(valid))
    sys.exit(1)


def _python_for(venv_python: Path) -> str:
    """Fallback ke python3 di PATH kalau venv belum di-setup."""
    return str(venv_python) if venv_python.exists() else "python3"


def run_step(label: str, cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n{'=' * 70}")
    print(f"STEP: {label}")
    print(f"  cwd: {cwd}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 70)

    if dry_run:
        print("  [dry-run] skipped")
        return

    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ STEP GAGAL: {label} (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)

    print(f"✅ {label} selesai ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator pipeline data ESG: NAS → chunk → Qdrant → sync companies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sector", type=str, default=None,
                         help="Proses hanya 1 sektor. Terima nama NAS (mis. 'Perbankan') atau "
                              "nama Qdrant (mis. 'Finance') — otomatis dikonversi untuk tiap step.")
    parser.add_argument("--skip-nas", action="store_true",
                         help="Lewati download+chunk dari NAS (pakai chunked_data/ yang sudah ada)")
    parser.add_argument("--skip-excel", action="store_true",
                         help="Lewati insert data Excel (esg_data_reports)")
    parser.add_argument("--skip-sync", action="store_true",
                         help="Lewati sync_companies.py di akhir")
    parser.add_argument("--only", choices=["nas", "insert", "excel", "sync"], default=None,
                         help="Jalankan hanya 1 step tertentu, abaikan flag --skip-*")
    parser.add_argument("--dry-run", action="store_true",
                         help="Tampilkan rencana eksekusi tanpa menjalankan apapun")
    args = parser.parse_args()

    run_nas = args.only in (None, "nas") and not args.skip_nas
    run_insert = args.only in (None, "insert")
    run_excel = args.only in (None, "excel") and not args.skip_excel
    run_sync = args.only in (None, "sync") and not args.skip_sync

    sector_nas = sector_qdrant = None
    if args.sector:
        sector_nas, sector_qdrant = resolve_sector_names(args.sector)
        if sector_nas != sector_qdrant:
            print(f"Sektor '{args.sector}' → NAS: '{sector_nas}' / Qdrant: '{sector_qdrant}'")

    pipeline_start = time.time()

    # ── Step 1: Download + chunk dari NAS ───────────────────────────────────
    if run_nas:
        cmd = [_python_for(PY_ESG_RAG_PYTHON), "dms_chunker.py"]
        if sector_nas:
            cmd += ["--sector", sector_nas]
        run_step("1/4 Download & chunk dokumen dari NAS", cmd, DATA_DIR, args.dry_run)
    else:
        print("\n(Step 1 dilewati: --skip-nas atau --only lain)")

    # ── Step 2: Insert chunk PDF ke Qdrant ──────────────────────────────────
    if run_insert:
        cmd = [_python_for(PY_ESG_RAG_PYTHON), "insertdata.py"]
        if sector_qdrant:
            cmd += ["--sector", sector_qdrant]
        run_step("2/4 Insert chunk PDF ke Qdrant (esg_reports)", cmd, QDRANT_INSERT_DIR, args.dry_run)
    else:
        print("\n(Step 2 dilewati: --only lain)")

    # ── Step 3: Insert chunk Excel ke Qdrant ────────────────────────────────
    if run_excel:
        cmd = [_python_for(PY_ESG_RAG_PYTHON), "insertdata_excel.py"]
        run_step("3/4 Insert chunk Excel ke Qdrant (esg_data_reports)", cmd, QDRANT_INSERT_DIR, args.dry_run)
    else:
        print("\n(Step 3 dilewati: --skip-excel atau --only lain)")

    # ── Step 4: Sync collection esg_companies ───────────────────────────────
    if run_sync:
        cmd = [_python_for(CHATBOT_API_PYTHON), "sync_companies.py"]
        run_step("4/4 Sync collection esg_companies", cmd, CHATBOT_API_DIR, args.dry_run)
        if not args.dry_run:
            print("\n⚠️  Jangan lupa restart py-esg-rag-chatbot-api supaya daftar")
            print("    perusahaan & system prompt ter-reload (uvicorn main:app --reload).")
    else:
        print("\n(Step 4 dilewati: --skip-sync atau --only lain)")

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE SELESAI dalam {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
