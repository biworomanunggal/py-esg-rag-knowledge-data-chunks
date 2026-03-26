#!/usr/bin/env python3
"""
Insert Excel ESG Data to Qdrant
================================
Script untuk menginsert data ESG dari chunked_data_excel ke Qdrant collection.
Collection name diambil dari environment variable QDRANT_COLLECTION_DATA_NAME.
"""

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_DATA_NAME", "esg_data_reports")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))

# Data directory
CHUNKED_DATA_DIR = PROJECT_ROOT / "chunked_data_excel"


def generate_point_id(content: str, nama_perusahaan: str, sumber_file: str) -> str:
    """Generate a unique hash ID based on content and metadata."""
    unique_string = f"{content}|{nama_perusahaan}|{sumber_file}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def get_existing_ids(client: QdrantClient, collection_name: str) -> set:
    """Get all existing point IDs from the collection."""
    existing_ids = set()
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            points, offset = result
            for point in points:
                existing_ids.add(point.id)
            if offset is None:
                break
    except Exception as e:
        print(f"Error fetching existing IDs: {e}")
    return existing_ids


def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int):
    """Create collection if it doesn't exist."""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")


def load_chunked_data() -> list:
    """Load all JSON files from chunked_data_excel directory."""
    all_chunks = []

    if not CHUNKED_DATA_DIR.exists():
        print(f"Directory not found: {CHUNKED_DATA_DIR}")
        return all_chunks

    for json_file in CHUNKED_DATA_DIR.glob("*.json"):
        print(f"Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def main():
    print("=" * 60)
    print("  INSERT EXCEL ESG DATA TO QDRANT")
    print("=" * 60)

    print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nTarget collection: {COLLECTION_NAME}")
    ensure_collection_exists(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    print("\nFetching existing IDs from collection...")
    existing_ids = get_existing_ids(client, COLLECTION_NAME)
    print(f"Found {len(existing_ids)} existing points")

    chunks = load_chunked_data()

    if not chunks:
        print("No chunks to insert")
        return

    points_to_insert = []
    skipped_count = 0

    print("\nProcessing chunks...")
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        nama_perusahaan = chunk.get("nama_perusahaan", "")
        sumber_file = chunk.get("sumber_file", "")
        nama_file = chunk.get("nama_file", "")
        sector = chunk.get("sector", "")
        metadata = chunk.get("metadata", {})

        point_id = generate_point_id(content, nama_perusahaan, sumber_file)

        if point_id in existing_ids:
            skipped_count += 1
            continue

        # Generate embedding with E5 prefix
        embedding = model.encode(f"passage: {content}").tolist()

        payload = {
            "content": content,
            "nama_perusahaan": nama_perusahaan,
            "sumber_file": sumber_file,
            "nama_file": nama_file,
            "sector": sector,
            "metadata": metadata
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        points_to_insert.append(point)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")

    print(f"\nSkipped {skipped_count} existing chunks")
    print(f"New chunks to insert: {len(points_to_insert)}")

    if points_to_insert:
        batch_size = 50
        total_batches = (len(points_to_insert) + batch_size - 1) // batch_size

        for i in range(0, len(points_to_insert), batch_size):
            batch = points_to_insert[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch_num = i // batch_size + 1
            print(f"  Inserted batch {batch_num}/{total_batches}: {len(batch)} points")

        print(f"\nSuccessfully inserted {len(points_to_insert)} new points")
    else:
        print("\nNo new data to insert")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks processed: {len(chunks)}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Inserted: {len(points_to_insert)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
