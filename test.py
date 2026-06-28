# # save as check_search.py and run: python check_search.py
# from dotenv import load_dotenv
# import os
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer

# load_dotenv()

# qdrant = QdrantClient(path = "./qdrant_data")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# query = "what courses are available"
# vector = embedder.encode(query, convert_to_numpy=True).tolist()
# print(f"Vector dim: {len(vector)}")

# results = qdrant.query_points(
#     collection_name="kayfa_paid_educational_tracks",
#     query=vector,
#     limit=3,
#     with_payload=True,
# ).points

# print(f"Results: {len(results)}")
# for r in results:
#     print(f"  score={r.score:.4f}  payload keys={list(r.payload.keys())}")
#     print(f"  preview: {str(r.payload)[:200]}")
# qdrant.close()






# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# import os
# from dotenv import load_dotenv
# load_dotenv()
# qdrant = QdrantClient(
#     # url = os.getenv("QDRANT_URL"),
#     # api_key = os.getenv("QDRANT_API_KEY")
#     path = "./qdrant_data"
# )
# for collection in qdrant.get_collections().collections:
#     print(f"Collection name: {collection.name}")
    # results = qdrant.query_points(
    #     collection_name=collection.name,
    #     query=[0.0] * collection.vectors_config.size,
    #     limit=1,
    # )
    # qdrant.delete_collection(collection.name)
    # print(f"  Vectors config: {collection.vectors_config}")
    # print(f"  Points count: {collection.points_count}")
    # print(f"  Shard number: {collection.shard_number}")
    # print(f"  Optimizers config: {collection.optimizers_config}")
#     from qdrant_client.models import Filter

#     points, _ = qdrant.scroll(
#         collection_name=collection.name,
#         limit=1,
#         with_payload=True,
#         with_vectors=False,
#     )

#     if points:
#         print("Payload schema:")
#         for key, value in points[0].payload.items():
#             print(f"{key}: {type(value).__name__}")
# qdrant.close()

# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os
# load_dotenv()
# url = os.getenv("MONGODB_URI")

# client = MongoClient(url)

# databases = client.list_database_names()

# keep = ["student_analytics"]

# for db in databases:
#     if db not in keep:
#         client.drop_database(db)
#         print(f"Dropped database: {db}")
#     else:
#         print(f"Kept database: {db}")

# delete the collections in Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()
qdrant = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY")
)
for collection in qdrant.get_collections().collections:
    print(f"Dropped collection: {collection.name}")
    qdrant.delete_collection(collection.name)