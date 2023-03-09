from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = "OPENAI_API_KEY"
    chroma_path: str = "data/chroma"
    openai_embedding_model: str = "text-embedding-ada-002"
    resume_path: str = "data/df.resumes.gzip"
    openai_collection_name: str = "openai"

    class Config:
        env_file = ".env"
