from config.config import cfg
from langchain_community.vectorstores import FAISS


class EmbeddingLoader:
    """
    A class to load embeddings from a FAISS vector store.

    Attributes:
        faiss_persist_directory (str): Directory where the FAISS index is persisted.
        embeddings (list): List of embeddings to be loaded.

    Methods:
        load_embeddings(): Loads the embeddings from the FAISS store.
    """

    def __init__(self, faiss_persist_directory, embeddings):
        """
        Constructs all the necessary attributes for the EmbeddingLoader object.

        Parameters:
            faiss_persist_directory (str): Directory where the FAISS index is persisted.
            embeddings (list): List of embeddings to be loaded.
        """
        self.faiss_persist_directory = faiss_persist_directory
        self.embeddings = embeddings

    def load_embeddings(self):
        """
        Loads the embeddings from the FAISS store and returns the FAISS database instance.

        Returns:
            FAISS: The loaded FAISS database instance.
        """
        # Load the FAISS database from the local filesystem
        new_db = FAISS.load_local(self.faiss_persist_directory, self.embeddings)
        return new_db

# Create an instance of EmbeddingLoader with configuration settings
embed_loader = EmbeddingLoader(cfg.faiss_persist_directory, cfg.embeddings)