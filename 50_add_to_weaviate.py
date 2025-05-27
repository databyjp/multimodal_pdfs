from helpers import WEAVIATE_COLLECTION_NAME
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.init import Auth
import base64
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

load_dotenv()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["APP_WEAVIATE_CLOUD_URL"],
    auth_credentials=Auth.api_key(os.environ["APP_WEAVIATE_CLOUD_APIKEY"]),
)

client.collections.delete(name=WEAVIATE_COLLECTION_NAME)

pdfs = client.collections.create(
    name=WEAVIATE_COLLECTION_NAME,
    properties=[
        Property(name="filepath", data_type=DataType.TEXT),
        Property(name="image", data_type=DataType.BLOB),
    ],
    vectorizer_config=[
        Configure.NamedVectors.none(
            name="colpali",  # Uses colpali-1.3
            vector_index_config=Configure.VectorIndex.hnsw(
                multi_vector=Configure.VectorIndex.MultiVector.multi_vector(),
                quantizer=Configure.VectorIndex.Quantizer.sq()
            )
        )
    ],
)


embedding_dir = Path("data/embeddings")
embedding_paths = embedding_dir.glob("*.npz")


with pdfs.batch.fixed_size(10) as batch:
    for embedding_path in embedding_paths:
        print(f"Importing files from: {embedding_path}")
        embeddings = np.load(embedding_path)
        for i, embedding in tqdm(enumerate(embeddings["embeddings"])):
            img_filepath_str = embeddings["filepaths"][i]
            img_path = Path(img_filepath_str)

            batch.add_object(
                properties={
                    "filepath": img_filepath_str,
                    "image": base64.b64encode(img_path.read_bytes()).decode("utf-8"),
                },
                vector={"colpali": embedding}
            )

if pdfs.batch.failed_objects:
    print(pdfs.batch.failed_objects[0].message)

print(len(pdfs))

client.close()
