import pandas as pd
import config
from loguru import logger

from db.core import client, embedding_function


logger.remove(0)
logger.add("scripts.log", format="{time} {level} {message}", level="INFO")
logger.info("initiating script")

logger.info("loading resumes")
df_resume = pd.read_parquet(config.Settings().resume_path)
logger.info("resumes loaded")
corpus, ids, categories = (
    list(df_resume["Resume_str"].values[0:5]),
    list(df_resume["ID"].values[0:5].astype(str)),
    list(df_resume["Category"].values[0:5]),
)
categories = [{"category": category} for category in categories]
logger.info(f"{categories}")

logger.info("deleting existing collection")

client.delete_collection(name=config.Settings().openai_collection_name)

logger.info(
    f"creating new collection named {config.Settings().openai_collection_name}"
)

collection = client.create_collection(
    name=config.Settings().openai_collection_name,
    embedding_function=embedding_function,
)

logger.info(
    f"adding documents to collection {config.Settings().openai_collection_name}"
)

collection.add(documents=corpus, ids=ids, metadatas=categories)
