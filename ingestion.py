from pathlib import Path
from qdrant_client.models import PointStruct
import uuid
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
import shutil
# shutil.rmtree("qdrant_data", ignore_errors=True)
def extract_sections(md_path: str):
    text = Path(md_path).read_text(encoding="utf-8")

    pattern = r"^(#{1,6}\s+.+)$"

    lines = text.splitlines()

    sections = []

    current_header = None
    current_content = []

    for line in lines:
        if re.match(pattern, line):
            if current_header:
                sections.append(
                    {
                        "header": current_header,
                        "content": "\n".join(current_content).strip(),
                    }
                )

            current_header = re.sub(r"^#+\s*", "", line).strip()
            current_content = []

        else:
            current_content.append(line)

    if current_header:
        sections.append(
            {
                "header": current_header,
                "content": "\n".join(current_content).strip(),
            }
        )

    return sections

model = SentenceTransformer(os.getenv("embedding_model"))


def embed_section(section):
    text = (
        f"{section['header']}\n\n"
        f"{section['content']}"
    )

    return model.encode(text).tolist()
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    # path="./qdrant_data",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def create_collection(collection_name, vector_size):
    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

def ingest_markdown(md_file):
    sections = extract_sections(md_file)

    collection_name = (
        Path(md_file).stem
        .lower()
        .replace(" ", "_")
    )

    sample_vector = embed_section(sections[0])

    create_collection(
        collection_name,
        len(sample_vector)
    )

    points = []

    for section in sections:
        vector = embed_section(section)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "header": section["header"],
                    "content": section["content"],
                    "source_file": Path(md_file).name,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"Successfully ingested {md_file} into collection '{collection_name}'.")
    
def ingest_courses(json_file):
    import json
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    collection_name = (
        Path(json_file).stem
        .lower()
        .replace(" ", "_")
    )

    sample_vector = embed_section(data[0])

    create_collection(
        collection_name,
        len(sample_vector)
    )

    points = []

    for section in data:
        vector = embed_section(section)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "header": section["header"],
                    "content": section["content"],
                    "source_file": Path(json_file).name,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"Successfully ingested {json_file} into collection '{collection_name}'.")
dir = r"data/text"
for filename in os.listdir(dir):
    if filename.endswith(".md"):
        print(f"Ingesting {filename}...")
        md_file = os.path.join(dir, filename)
        ingest_markdown(md_file)
