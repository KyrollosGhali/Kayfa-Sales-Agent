import json
from pathlib import Path
import re
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client.models import PointStruct
import re

def parse_markdown(md_text: str):
    sections = []
    matches = list(
        re.finditer(
            r"^\s*(#{1,6})\s+(.+?)\s*$",
            md_text,
            re.MULTILINE
        )
    )
    
    print(f"Found {len(matches)} headers in the markdown text.")
    for i, match in enumerate(matches):
        level = len(match.group(1))
        header = match.group(2).strip()

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(md_text)
        )

        content = md_text[start:end].strip()

        sections.append(
            {
                "level": level,
                "header": header,
                "content": content
            }
        )

    return sections

markdown_files = os.listdir("data/text")

qdrant = QdrantClient(
    path = "./qdrant_data"
)
if qdrant.collection_exists("kayfa_knowledge"):
    qdrant.delete_collection("kayfa_knowledge")
    qdrant.recreate_collection(
        collection_name="kayfa_knowledge",
        vectors_config={
            "size": 384,
            "distance": "Cosine"
        }
    )
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

for md_file in markdown_files:
    print(f"Processing file: {md_file}")
    if "diploma" in md_file.lower() or any(policy in md_file.lower() for policy in ["policies", "privacy", "overview"]):
        json_file = os.path.join("data/text", md_file)
        with open(os.path.join("data/text", md_file), "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Extracted text from {md_file}:")
        print(text[:500])  # Print the first 500 characters of the text for verification
        sections = parse_markdown(text)
        print(sections)
        with open(
            f"data/json/{md_file.replace('.md', '')}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                sections,
                f,
                indent=4,
                ensure_ascii=False
            )
        for idx, section in enumerate(sections):

            embedding_text = f"""
            {section['header']}

            {section['content']}
            """

            vector = embedding_model.encode(
                embedding_text
            ).tolist()

            payload = {
                "document": md_file.replace(".md", "").replace("_", " "),
                "header": section["header"],
                "content": section["content"],
                "type": "policy"
            }
            collection_name = "kayfa_knowledge"
            if not qdrant.collection_exists(collection_name):
                qdrant.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": len(vector),
                        "distance": "Cosine"
                    }
                )
            qdrant.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload
                    )
                ]
            )
qdrant.close()