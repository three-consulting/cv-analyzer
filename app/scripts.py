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
resumes, jobs, skills, ids = (
    list(df_resume["resumes"].values),
    list(df_resume["jobs"].values),
    list(df_resume["skills"].values),
    list(range(df_resume["resumes"].size)),
)
categories = [{"job": job, "skill": skill } for job, skill in zip(jobs, skills)]
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

collection.add(documents=resumes, ids=ids, metadatas=categories)
