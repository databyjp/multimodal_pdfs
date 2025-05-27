from helpers import WEAVIATE_COLLECTION_NAME, get_model_and_processor, text_to_colpali
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

load_dotenv()

model, processor = get_model_and_processor()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["APP_WEAVIATE_CLOUD_URL"],
    auth_credentials=Auth.api_key(os.environ["APP_WEAVIATE_CLOUD_APIKEY"]),
)

pdfs = client.collections.get(name=WEAVIATE_COLLECTION_NAME)

queries = [
    "Diagrams of Weaviate cluster architecture, with shards, indexes and model integrations.",
    "How much does vector quantization impact memory footprint?",
    "Can I use natural language to modify Weaviate data?",
]

for i, query in enumerate(queries):
    print(f"Query: {queries[i]}")
    query_vector = text_to_colpali([query], model=model, processor=processor)[0]

    r = pdfs.query.near_vector(
        near_vector=query_vector,
        target_vector="colpali",
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in r.objects:
        print(o.properties)
        print(o.metadata.distance)

client.close()
