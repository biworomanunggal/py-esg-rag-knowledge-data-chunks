# Pipeline py-esg-rag

Urutan end-to-end dari data mentah sampai chatbot siap digunakan.

## Step 1: Siapkan Dokumen Sumber

Taruh file PDF (Annual Report / Sustainability Report) ke folder `resource/` sesuai sektor:

```
resource/
├── Finance/        → Bank Jago, OCBC NISP, dll
├── Mining/         → Adaro, Archi, dll
├── Energy/         → Elnusa, Apexindo, dll
├── Plantation/     → Astra Agro, dll
├── Transportation/ → AirAsia, Adi Sarana, dll
├── F&B/            → Delta Djakarta, dll
├── Chemicals/      → Barito Pacific, dll
└── Hostpitality/   → MNC Land, dll
```

Untuk data Excel (indikator ESG kuantitatif), taruh di `excel_resources/`.

## Step 2: Chunking — Pecah Dokumen Jadi Potongan Kecil

**PDF** → jalankan `chunks/pdf_chunker.py`

- Ekstrak teks & tabel dari PDF via `pdfplumber`
- Pecah per section/heading (~2000 karakter, 200 overlap)
- Simpan metadata (halaman, section, nama perusahaan)
- Output: file JSON di `chunked_data/` (contoh: `pt_bank_jago_ir_2024_chunks.json`)

**Excel** → jalankan `chunks/excel_chunker.py`

- Parse struktur pivot Excel (kolom = perusahaan)
- Konversi data tabular ke teks semantik
- Output: file JSON di `chunked_data_excel/`

## Step 3: Embedding + Insert ke Qdrant

Jalankan `qdrant_insert/insertdata.py` (untuk PDF) dan `qdrant_insert/insertdata_excel.py` (untuk Excel):

1. Baca file JSON dari `chunked_data/` atau `chunked_data_excel/`
2. Generate embedding vektor menggunakan `intfloat/multilingual-e5-base` (768 dimensi)
3. Upsert ke Qdrant dengan deduplikasi (MD5 hash)
4. **Dua collection terpisah:**
   - `esg_reports` — chunk dari PDF
   - `esg_data_reports` — chunk dari Excel

## Step 4: Chatbot / API Siap Digunakan

**Opsi A — CLI Chatbot:** `chatbot/chatbot_canggih.py`

**Opsi B — REST API (production):** `api/main.py` + `api/chatbot_service.py`

Alur saat user bertanya:

```
User Query
  ↓
1. Analisis query → deteksi perusahaan, topik ESG, sektor
  ↓
2. Cari di Qdrant (kedua collection)
   - Semantic search (embedding similarity)
   - Filter by perusahaan/sektor jika terdeteksi
  ↓
3. Reranking → skor multi-faktor (similarity + keyword + topik + perusahaan)
  ↓
4. Format konteks dari hasil retrieval
  ↓
5. Kirim ke LLM (Model Ark) dengan system prompt + history
  ↓
6. Return response + metadata (token usage, sources, analysis)
```

## Ringkasan Visual

```
PDF/Excel  →  Chunker  →  JSON  →  Embedding + Qdrant Insert  →  Chatbot/API
 (Step 1)     (Step 2)   (Step 2)        (Step 3)                  (Step 4)
```
