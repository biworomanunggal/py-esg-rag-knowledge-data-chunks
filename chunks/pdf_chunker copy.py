#!/usr/bin/env python3
"""
PDF Chunker dengan Table Extraction untuk ESG Reports
======================================================
Script untuk mengekstrak teks dan tabel dari PDF laporan keberlanjutan/tahunan
dan menghasilkan chunks dalam format yang kompatibel dengan Qdrant.

Features:
- Ekstraksi teks dengan layout preservation
- Ekstraksi tabel dengan anotasi
- Smart chunking berdasarkan section/heading
- Metadata extraction (halaman, section, subsection)
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system("pip install pdfplumber")
    import pdfplumber

try:
    from tabulate import tabulate
except ImportError:
    print("Installing tabulate...")
    os.system("pip install tabulate")
    from tabulate import tabulate


# Configuration
CHUNK_SIZE = 2000  # Target karakter per chunk
CHUNK_OVERLAP = 200  # Overlap antar chunk
MIN_CHUNK_SIZE = 500  # Minimum karakter untuk chunk

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resource"
OUTPUT_DIR = PROJECT_DIR / "chunked_data"


@dataclass
class ChunkMetadata:
    """Metadata untuk setiap chunk."""
    section: str
    subsection: str
    page_range: str


@dataclass
class Chunk:
    """Representasi satu chunk dokumen."""
    id: int
    content: str
    nama_perusahaan: str
    sumber_file: str
    metadata: ChunkMetadata


class PDFChunker:
    """PDF Chunker dengan Table Extraction."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pdf_name = self.pdf_path.stem
        self.chunks: List[Chunk] = []
        self.current_section = "Umum"
        self.current_subsection = ""

        # Extract company name and report type from filename
        self.company_name, self.report_type = self._parse_filename()

    def _parse_filename(self) -> Tuple[str, str]:
        """Parse company name and report type from filename."""
        filename = self.pdf_name

        # Pattern: "Company Name_Report Type_Year"
        # Example: "PT Bank OCBC NISP Tbk_AR & SR_2024"
        parts = filename.split("_")

        if len(parts) >= 2:
            company = parts[0].strip()
            report_type = "_".join(parts[1:]).strip()

            # Determine report type description
            if "SR" in report_type.upper():
                report_desc = "Laporan Keberlanjutan"
            elif "AR" in report_type.upper():
                report_desc = "Laporan Tahunan"
            else:
                report_desc = "Laporan"

            # Extract year
            year_match = re.search(r"(\d{4})", report_type)
            year = year_match.group(1) if year_match else ""

            if year:
                report_desc = f"{report_desc} {year}"

            return company, report_desc
        else:
            return filename, "Laporan"

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect section headers from text."""
        # Common section patterns in Indonesian reports
        section_patterns = [
            # Indonesian
            r"^(Tentang\s+(?:Laporan|Perusahaan|Kami))",
            r"^(Profil\s+(?:Perusahaan|Singkat))",
            r"^(Ikhtisar\s+(?:Keuangan|Kinerja|Data))",
            r"^(Tata\s+Kelola\s+(?:Perusahaan)?)",
            r"^(Kinerja\s+(?:Keberlanjutan|Lingkungan|Sosial|Ekonomi))",
            r"^(Strategi\s+(?:Keberlanjutan|Bisnis))",
            r"^(Manajemen\s+Risiko)",
            r"^(Tanggung\s+Jawab\s+Sosial)",
            r"^(Sumber\s+Daya\s+Manusia)",
            r"^(Laporan\s+(?:Keuangan|Auditor))",
            r"^(Informasi\s+(?:Umum|Perusahaan))",
            # English
            r"^(About\s+(?:This\s+Report|The\s+Company|Us))",
            r"^(Company\s+Profile)",
            r"^(Financial\s+(?:Highlights|Summary))",
            r"^(Corporate\s+Governance)",
            r"^(Sustainability\s+(?:Performance|Report))",
            r"^(Environmental\s+(?:Performance|Impact))",
            r"^(Social\s+(?:Performance|Responsibility))",
            r"^(Human\s+(?:Resources|Capital))",
        ]

        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_tables(self, page) -> List[str]:
        """Extract tables from a page and format them."""
        tables = page.extract_tables()
        formatted_tables = []

        for table in tables:
            if not table or len(table) < 2:
                continue

            # Clean table data
            cleaned_table = []
            for row in table:
                cleaned_row = []
                has_content = False
                for cell in row:
                    if cell is None:
                        cleaned_row.append("")
                    else:
                        # Clean cell content
                        cell_text = str(cell).strip()
                        cell_text = re.sub(r'\s+', ' ', cell_text)
                        cleaned_row.append(cell_text)
                        if cell_text:
                            has_content = True
                # Only add row if it has some content
                if has_content:
                    cleaned_table.append(cleaned_row)

            # Skip tables with less than 2 rows after cleaning
            if len(cleaned_table) < 2:
                continue

            # Check if first row looks like a header (contains text, not just numbers)
            first_row = cleaned_table[0]
            has_header = any(
                cell and not cell.replace(',', '').replace('.', '').replace('-', '').replace(' ', '').isdigit()
                for cell in first_row if cell
            )

            if cleaned_table:
                try:
                    if has_header and len(cleaned_table) > 1:
                        # Format as markdown table with first row as header
                        table_str = tabulate(cleaned_table[1:], headers=cleaned_table[0], tablefmt="pipe")
                    else:
                        # No clear header, format all rows as grid
                        table_str = tabulate(cleaned_table, tablefmt="simple")
                    formatted_tables.append(f"\n[TABEL]\n{table_str}\n[/TABEL]\n")
                except Exception:
                    # Fallback: simple format
                    table_lines = []
                    for row in cleaned_table:
                        table_lines.append(" | ".join(row))
                    formatted_tables.append("\n[TABEL]\n" + "\n".join(table_lines) + "\n[/TABEL]\n")

        return formatted_tables

    def _extract_page_content(self, page, page_num: int) -> Tuple[str, List[str]]:
        """Extract text and tables from a single page."""
        # Extract text
        text = page.extract_text() or ""

        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces

        # Extract tables
        tables = self._extract_tables(page)

        return text, tables

    def _smart_chunk(self, text: str, start_page: int, end_page: int) -> List[Dict]:
        """Smart chunking based on content."""
        chunks = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n{2,}', text)

        current_chunk = ""
        chunk_start_page = start_page

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for section header
            detected_section = self._detect_section(para)
            if detected_section:
                # Save current chunk if exists
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page_range": f"{chunk_start_page}-{end_page}" if chunk_start_page != end_page else str(chunk_start_page),
                        "section": self.current_section,
                        "subsection": self.current_subsection
                    })
                    current_chunk = ""
                    chunk_start_page = end_page

                self.current_section = detected_section
                self.current_subsection = ""

            # Add paragraph to current chunk
            if len(current_chunk) + len(para) > CHUNK_SIZE:
                # Save current chunk
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page_range": f"{chunk_start_page}-{end_page}" if chunk_start_page != end_page else str(chunk_start_page),
                        "section": self.current_section,
                        "subsection": self.current_subsection
                    })

                # Start new chunk with overlap
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + "\n" + para
                chunk_start_page = end_page
            else:
                current_chunk += "\n" + para

        # Don't forget last chunk
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append({
                "content": current_chunk.strip(),
                "page_range": f"{chunk_start_page}-{end_page}" if chunk_start_page != end_page else str(chunk_start_page),
                "section": self.current_section,
                "subsection": self.current_subsection
            })

        return chunks

    def process(self) -> List[Dict]:
        """Process the PDF and generate chunks."""
        print(f"\n{'='*60}")
        print(f"Processing: {self.pdf_path.name}")
        print(f"Company: {self.company_name}")
        print(f"Report: {self.report_type}")
        print(f"{'='*60}")

        all_chunks = []

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            # Process pages in batches for better context
            batch_size = 3  # Process 3 pages at a time
            accumulated_text = ""
            batch_start_page = 1

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                print(f"  Processing page {page_num}/{total_pages}...", end="\r")

                # Extract content
                text, tables = self._extract_page_content(page, page_num)

                # Combine text and tables
                page_content = text
                for table in tables:
                    page_content += "\n" + table

                accumulated_text += "\n\n" + page_content

                # Process batch
                if (i + 1) % batch_size == 0 or i == total_pages - 1:
                    batch_chunks = self._smart_chunk(
                        accumulated_text,
                        batch_start_page,
                        page_num
                    )
                    all_chunks.extend(batch_chunks)

                    # Reset for next batch (with overlap)
                    accumulated_text = accumulated_text[-500:] if len(accumulated_text) > 500 else ""
                    batch_start_page = page_num

            print(f"\n  Generated {len(all_chunks)} chunks")

        # Format final chunks
        final_chunks = []
        for idx, chunk_data in enumerate(all_chunks, 1):
            final_chunks.append({
                "id": idx,
                "content": chunk_data["content"],
                "nama_perusahaan": self.company_name,
                "sumber_file": self.report_type,
                "metadata": {
                    "section": chunk_data["section"],
                    "subsection": chunk_data["subsection"],
                    "page_range": chunk_data["page_range"]
                }
            })

        self.chunks = final_chunks
        return final_chunks

    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save chunks to JSON file."""
        if output_path is None:
            # Generate output filename
            safe_name = re.sub(r'[^\w\-_]', '_', self.pdf_name.lower())
            output_path = OUTPUT_DIR / f"{safe_name}_chunks.json"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved to: {output_path}")
        return output_path


def process_all_pdfs(resource_dir: Path = RESOURCE_DIR) -> Dict[str, Path]:
    """Process all PDFs in resource directory."""
    results = {}

    if not resource_dir.exists():
        print(f"Resource directory not found: {resource_dir}")
        return results

    pdf_files = list(resource_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {resource_dir}")
        return results

    print(f"\nFound {len(pdf_files)} PDF files to process")
    print("="*60)

    for pdf_path in pdf_files:
        try:
            chunker = PDFChunker(str(pdf_path))
            chunks = chunker.process()
            output_path = chunker.save()
            results[pdf_path.name] = output_path

            print(f"  Chunks: {len(chunks)}")
            print()

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()

    return results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Chunker untuk ESG Reports")
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path ke file PDF spesifik (opsional, default: process semua PDF di resource/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path output file JSON (opsional)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PDF CHUNKER - ESG Report Processor")
    print("="*60)

    if args.pdf:
        # Process single PDF
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}")
            return

        chunker = PDFChunker(str(pdf_path))
        chunks = chunker.process()

        output_path = Path(args.output) if args.output else None
        chunker.save(output_path)

        print(f"\nTotal chunks: {len(chunks)}")

    else:
        # Process all PDFs in resource directory
        results = process_all_pdfs()

        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for pdf_name, output_path in results.items():
            print(f"  {pdf_name} -> {output_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
