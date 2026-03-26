import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))

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
    """Load all JSON files from chunked_data directory."""
    all_chunks = []

    if not CHUNKED_DATA_DIR.exists():
        print(f"Directory not found: {CHUNKED_DATA_DIR}")
        return all_chunks

    for json_file in CHUNKED_DATA_DIR.glob("**/*.json"):
        print(f"Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def main():
    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    ensure_collection_exists(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    print("Fetching existing IDs from collection...")
    existing_ids = get_existing_ids(client, COLLECTION_NAME)
    print(f"Found {len(existing_ids)} existing points")

    chunks = load_chunked_data()

    if not chunks:
        print("No chunks to insert")
        return

    points_to_insert = []
    skipped_count = 0

    for chunk in chunks:
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

        embedding = model.encode(content).tolist()

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

    print(f"Skipped {skipped_count} existing chunks")
    print(f"New chunks to insert: {len(points_to_insert)}")

    if points_to_insert:
        batch_size = 100
        for i in range(0, len(points_to_insert), batch_size):
            batch = points_to_insert[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"Inserted batch {i // batch_size + 1}: {len(batch)} points")

        print(f"Successfully inserted {len(points_to_insert)} new points")
    else:
        print("No new data to insert")


if __name__ == "__main__":
    main()
