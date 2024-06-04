import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class Config:
    """
    Configuration class for initializing the language model and embeddings settings.
    """

    def __init__(self):
        self.directory_path = 'all_files'
        self.path_of_file_to_be_deleted = "test_files\Hackathon.pdf" # Test to delete a file
        # FAISS Store Configuration
        self.faiss_persist_directory = Path(os.getenv('FAISS_STORE', 'faiss_store'))
        self.faiss_persist_directory.mkdir(exist_ok=True)
        
        # Embedding and AI model parameters
        self.embedding_model = 'text-embedding-3-small' # You can use text-embedding-ada-002 or text-embedding-3-large
        self.model = 'gpt-3.5-turbo'
        self.temperature = '0.2'
        self.embeddings = OpenAIEmbeddings(
            chunk_size=300, 
            model=self.embedding_model, 
            show_progress_bar=True
        )
        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=self.temperature,
        )

cfg = Config()
