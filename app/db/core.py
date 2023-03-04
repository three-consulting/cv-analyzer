import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from app.config import Settings as AppSettings

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet", persist_directory=AppSettings().chroma_path
    )
)

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AppSettings().openai_api_key,
    model_name=AppSettings().openai_embedding_model,
)
