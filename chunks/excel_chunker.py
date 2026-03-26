#!/usr/bin/env python3
"""
Excel ESG Data Chunker untuk RAG System
=========================================
Script untuk mengekstrak dan chunk data ESG dari spreadsheet Excel
dengan format pivoted table (company-wise columns).

Features:
- Parse pivoted Excel structure dengan multiple companies
- Generate semantic text dari tabular data
- Group by Company + Category (Lingkungan/Sosial/Tata Kelola/General)
- Rich metadata untuk hybrid search
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    os.system("pip install pandas openpyxl")
    import pandas as pd


# Configuration
YEARS = [2021, 2022, 2023, 2024]
KINERJA_CATEGORIES = ["Lingkungan", "Sosial", "Tata Kelola", "General"]

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
EXCEL_DIR = PROJECT_DIR / "excel_resources"
OUTPUT_DIR = PROJECT_DIR / "chunked_data"


@dataclass
class ESGIndicator:
    """Single ESG indicator with values across years."""
    kinerja: str
    topik: str
    sub_topik: str
    indikator: str
    satuan: str
    kategori: str
    klasifikasi: str
    indikator_kunci: str
    values: Dict[int, any]  # year -> value


@dataclass
class CompanyESGData:
    """ESG data for a single company."""
    nama_perusahaan: str
    indicators: List[ESGIndicator]


class ExcelESGChunker:
    """Excel ESG Data Chunker."""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.excel_name = self.excel_path.stem
        self.companies: List[str] = []
        self.company_col_map: Dict[str, int] = {}  # company -> start column index
        self.indicators_metadata: List[Dict] = []
        self.chunks: List[Dict] = []

    def _parse_header_structure(self, df: pd.DataFrame) -> None:
        """Parse the header rows to extract company names and column mapping."""
        # Row 0 contains company names (every 4 columns starting from col 9)
        # Row 1 contains years (2021, 2022, 2023, 2024 repeated)

        self.companies = []
        self.company_col_map = {}

        for col_idx in range(9, df.shape[1], 4):
            company_name = df.iloc[0, col_idx]
            if pd.notna(company_name):
                company_name = str(company_name).strip()
                # Skip "Rata-rata Indikator" (average column)
                if "rata-rata" not in company_name.lower():
                    self.companies.append(company_name)
                    self.company_col_map[company_name] = col_idx

        print(f"Found {len(self.companies)} companies")

    def _parse_indicators(self, df: pd.DataFrame) -> List[Dict]:
        """Parse indicator metadata from rows 2+."""
        indicators = []

        # Track current kinerja, topik, sub_topik for filling in merged cells
        current_kinerja = ""
        current_topik = ""
        current_sub_topik = ""

        # Start from row 3 (index 2 is header, data starts at 3)
        for row_idx in range(3, df.shape[0]):
            row = df.iloc[row_idx]

            # Update tracking variables if cell is not NaN
            if pd.notna(row.iloc[0]):
                current_kinerja = str(row.iloc[0]).strip()
            if pd.notna(row.iloc[1]):
                current_topik = str(row.iloc[1]).strip()
            if pd.notna(row.iloc[2]):
                current_sub_topik = str(row.iloc[2]).strip()

            indikator = row.iloc[3] if pd.notna(row.iloc[3]) else ""
            satuan = row.iloc[4] if pd.notna(row.iloc[4]) else ""
            kategori = row.iloc[5] if pd.notna(row.iloc[5]) else ""
            klasifikasi = row.iloc[6] if pd.notna(row.iloc[6]) else ""
            indikator_kunci = row.iloc[7] if pd.notna(row.iloc[7]) else ""

            # Skip rows without indicator name
            if not indikator:
                continue

            indicators.append({
                "row_idx": row_idx,
                "kinerja": current_kinerja,
                "topik": current_topik,
                "sub_topik": current_sub_topik,
                "indikator": str(indikator).strip(),
                "satuan": str(satuan).strip(),
                "kategori": str(kategori).strip(),
                "klasifikasi": str(klasifikasi).strip(),
                "indikator_kunci": str(indikator_kunci).strip()
            })

        self.indicators_metadata = indicators
        print(f"Found {len(indicators)} indicators")
        return indicators

    def _get_company_values(self, df: pd.DataFrame, company: str, row_idx: int) -> Dict[int, any]:
        """Get values for a company across all years."""
        start_col = self.company_col_map[company]
        values = {}

        for i, year in enumerate(YEARS):
            col_idx = start_col + i
            if col_idx < df.shape[1]:
                val = df.iloc[row_idx, col_idx]
                if pd.notna(val):
                    # Clean and convert value
                    if isinstance(val, str):
                        val = val.strip()
                    values[year] = val
                else:
                    values[year] = None

        return values

    def _format_value(self, value: any, satuan: str) -> str:
        """Format value with unit for display."""
        if value is None:
            return "Tidak tersedia"

        # Format numbers with thousand separator
        if isinstance(value, (int, float)):
            if value == int(value):
                formatted = f"{int(value):,}".replace(",", ".")
            else:
                formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{formatted} {satuan}".strip()

        return f"{value} {satuan}".strip() if satuan else str(value)

    def _format_trend(self, values: Dict[int, any]) -> str:
        """Format trend across years."""
        trend_parts = []
        for year in YEARS:
            val = values.get(year)
            if val is not None:
                if isinstance(val, (int, float)):
                    if val == int(val):
                        formatted = f"{int(val):,}".replace(",", ".")
                    else:
                        formatted = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    trend_parts.append(f"{formatted} ({year})")
                else:
                    trend_parts.append(f"{val} ({year})")
            else:
                trend_parts.append(f"- ({year})")

        return " -> ".join(trend_parts)

    def _generate_semantic_content(self, company: str, kinerja: str,
                                   indicators: List[Tuple[Dict, Dict[int, any]]]) -> str:
        """Generate semantic text content for a chunk."""
        lines = []

        # Header
        lines.append(f"{company} - Kinerja {kinerja}")
        lines.append("")

        # Group indicators by topik and sub_topik
        current_topik = ""
        current_sub_topik = ""

        for indicator_meta, values in indicators:
            # Add topic header if changed
            if indicator_meta["topik"] != current_topik:
                current_topik = indicator_meta["topik"]
                current_sub_topik = ""
                lines.append(f"## {current_topik}")
                lines.append("")

            # Add sub-topic header if changed
            if indicator_meta["sub_topik"] != current_sub_topik:
                current_sub_topik = indicator_meta["sub_topik"]
                if current_sub_topik:
                    lines.append(f"### {current_sub_topik}")
                    lines.append("")

            # Add indicator with value
            indikator = indicator_meta["indikator"]
            satuan = indicator_meta["satuan"]

            # Get latest non-null value
            latest_value = None
            latest_year = None
            for year in reversed(YEARS):
                if values.get(year) is not None:
                    latest_value = values[year]
                    latest_year = year
                    break

            if latest_value is not None:
                formatted_value = self._format_value(latest_value, satuan)
                lines.append(f"- {indikator}: {formatted_value} ({latest_year})")

                # Add trend if multiple years have data
                non_null_count = sum(1 for v in values.values() if v is not None)
                if non_null_count > 1:
                    trend = self._format_trend(values)
                    lines.append(f"  Trend: {trend}")
            else:
                lines.append(f"- {indikator}: Data tidak tersedia")

            lines.append("")

        return "\n".join(lines)

    def _extract_company_data(self, df: pd.DataFrame, company: str) -> Dict[str, List[Tuple[Dict, Dict[int, any]]]]:
        """Extract all ESG data for a company, grouped by kinerja category."""
        data_by_kinerja = {k: [] for k in KINERJA_CATEGORIES}

        for indicator_meta in self.indicators_metadata:
            row_idx = indicator_meta["row_idx"]
            kinerja = indicator_meta["kinerja"]

            if kinerja not in data_by_kinerja:
                continue

            values = self._get_company_values(df, company, row_idx)

            # Only include if at least one value exists
            if any(v is not None for v in values.values()):
                data_by_kinerja[kinerja].append((indicator_meta, values))

        return data_by_kinerja

    def process(self) -> List[Dict]:
        """Process the Excel file and generate chunks."""
        print(f"\n{'='*60}")
        print(f"Processing: {self.excel_path.name}")
        print(f"{'='*60}")

        # Read Excel file
        df = pd.read_excel(self.excel_path, sheet_name="Finance", header=None)
        print(f"Excel shape: {df.shape}")

        # Parse structure
        self._parse_header_structure(df)
        self._parse_indicators(df)

        # Generate chunks for each company + category combination
        all_chunks = []
        chunk_id = 1

        for company in self.companies:
            print(f"  Processing: {company[:50]}...")
            company_data = self._extract_company_data(df, company)

            for kinerja in KINERJA_CATEGORIES:
                indicators = company_data.get(kinerja, [])

                if not indicators:
                    continue

                # Generate semantic content
                content = self._generate_semantic_content(company, kinerja, indicators)

                # Extract topics for metadata
                topics = list(set(ind[0]["topik"] for ind in indicators if ind[0]["topik"]))

                # Create chunk
                chunk = {
                    "id": chunk_id,
                    "content": content,
                    "nama_perusahaan": company,
                    "sumber_file": "Data Dashboard KESG Finance",
                    "nama_file": self.excel_path.name,
                    "sector": "Finance",
                    "metadata": {
                        "kinerja": kinerja,
                        "data_type": "tabular_esg",
                        "years": [str(y) for y in YEARS],
                        "topics": topics,
                        "indicator_count": len(indicators)
                    }
                }

                all_chunks.append(chunk)
                chunk_id += 1

        self.chunks = all_chunks
        print(f"\nGenerated {len(all_chunks)} chunks")
        return all_chunks

    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save chunks to JSON file."""
        if output_path is None:
            output_path = OUTPUT_DIR / "data_excel_kesg.json"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved to: {output_path}")
        return output_path


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Excel ESG Chunker untuk RAG System")
    parser.add_argument(
        "--excel",
        type=str,
        default=str(EXCEL_DIR / "Data Dashboard KESGI.xlsx"),
        help="Path ke file Excel"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path output file JSON (opsional)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  EXCEL ESG CHUNKER - Data Dashboard KESG")
    print("="*60)

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: File not found: {excel_path}")
        return

    chunker = ExcelESGChunker(str(excel_path))
    chunks = chunker.process()

    output_path = Path(args.output) if args.output else None
    chunker.save(output_path)

    # Print sample chunk
    if chunks:
        print("\n" + "="*60)
        print("  SAMPLE CHUNK (First)")
        print("="*60)
        sample = chunks[0]
        print(f"ID: {sample['id']}")
        print(f"Company: {sample['nama_perusahaan']}")
        print(f"Category: {sample['metadata']['kinerja']}")
        print(f"Topics: {sample['metadata']['topics']}")
        print(f"Indicators: {sample['metadata']['indicator_count']}")
        print(f"\nContent Preview (first 1000 chars):")
        print("-"*40)
        print(sample['content'][:1000])
        print("-"*40)

    print("\nDone!")


if __name__ == "__main__":
    main()
