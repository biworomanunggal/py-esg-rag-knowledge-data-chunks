import os
import json
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_TOKEN = os.getenv("QDRANT_TOKEN")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_TOKEN = os.getenv("EMBEDDING_API_TOKEN")

CHUNKED_DATA_DIR = Path(__file__).parent.parent / "chunked_data"


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


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts via the embedding API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_TOKEN}"
    }
    response = requests.post(
        f"{EMBEDDING_API_URL}/encode-batch",
        headers=headers,
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


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

def list_available_sectors() -> list:
    """List all available sector directories in chunked_data."""
    if not CHUNKED_DATA_DIR.exists():
        return []
    return sorted([d.name for d in CHUNKED_DATA_DIR.iterdir() if d.is_dir()])


def load_chunked_data(sector: str = None) -> list:
    """Load JSON files from chunked_data directory, optionally filtered by sector."""
    all_chunks = []

    if not CHUNKED_DATA_DIR.exists():
        print(f"Directory not found: {CHUNKED_DATA_DIR}")
        return all_chunks

    if sector:
        sector_dir = CHUNKED_DATA_DIR / sector
        if not sector_dir.exists():
            available = list_available_sectors()
            print(f"Sector '{sector}' not found. Available sectors: {', '.join(available)}")
            return all_chunks
        search_path = sector_dir
        print(f"Filtering by sector: {sector}")
    else:
        search_path = CHUNKED_DATA_DIR

    for json_file in search_path.glob("**/*.json"):
        print(f"Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Insert chunked data into Qdrant")
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Insert only a specific sector (e.g. --sector Energy). "
             "Case-sensitive, must match folder name in chunked_data/."
    )
    parser.add_argument(
        "--list-sectors",
        action="store_true",
        help="List available sectors and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_sectors:
        sectors = list_available_sectors()
        print("Available sectors:")
        for s in sectors:
            print(f"  - {s}")
        return

    if not EMBEDDING_API_URL or not EMBEDDING_API_TOKEN:
        print("Error: EMBEDDING_API_URL and EMBEDDING_API_TOKEN must be set in .env")
        return

    print("Connecting to Qdrant...")
    is_remote = "katadata.co.id" in QDRANT_HOST
    if is_remote:
        client = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_TOKEN)
        print(f"Using remote Qdrant: {QDRANT_HOST} (with API key)")
    else:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Using embedding API: {EMBEDDING_API_URL}")

    ensure_collection_exists(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    print("Fetching existing IDs from collection...")
    existing_ids = get_existing_ids(client, COLLECTION_NAME)
    print(f"Found {len(existing_ids)} existing points")

    chunks = load_chunked_data(sector=args.sector)

    if not chunks:
        print("No chunks to insert")
        return

    new_chunks = []
    skipped_count = 0

    for chunk in chunks:
        content = chunk.get("content", "")
        nama_perusahaan = chunk.get("nama_perusahaan", "")
        sumber_file = chunk.get("sumber_file", "")

        point_id = generate_point_id(content, nama_perusahaan, sumber_file)

        if point_id in existing_ids:
            skipped_count += 1
            continue

        new_chunks.append((point_id, chunk))

    print(f"Skipped {skipped_count} existing chunks")
    print(f"New chunks to process: {len(new_chunks)}")

    if not new_chunks:
        print("No new data to insert")
        return

    batch_size = 100
    total_inserted = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        texts = [item[1].get("content", "") for item in batch]

        print(f"Encoding batch {i // batch_size + 1} ({len(batch)} texts)...")
        embeddings = get_embeddings_batch(texts)

        points = []
        for (point_id, chunk), embedding in zip(batch, embeddings):
            payload = {
                "content": chunk.get("content", ""),
                "nama_perusahaan": chunk.get("nama_perusahaan", ""),
                "sumber_file": chunk.get("sumber_file", ""),
                "nama_file": chunk.get("nama_file", ""),
                "sector": chunk.get("sector", ""),
                "metadata": chunk.get("metadata", {})
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_inserted += len(points)
        print(f"Inserted batch {i // batch_size + 1}: {len(points)} points")

    print(f"Successfully inserted {total_inserted} new points")


if __name__ == "__main__":
    main()
