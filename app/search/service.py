import pandas as pd
from db.core import client, embedding_function
from config import Settings


def semantic_search(query: str, filter: str, k: int = 10):
    collection = client.get_collection(
        name=Settings().openai_collection_name,
        embedding_function=embedding_function,
    )

    if filter:
        res = collection.query(
            query_texts=[query],
            where_document={"$contains": filter},
            n_results=k,
        )
    else:
        res = collection.query(
            query_texts=[query],
            n_results=k,
        )

    result_df = pd.DataFrame(
        {
            "distances": list(res["distances"][0]),
            "documents": list(res["documents"][0]),
        }
    )

    return result_df
