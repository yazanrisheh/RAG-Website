from config.config import cfg
from doc_process.doc_splitter import doc_processor
from langchain_community.vectorstores import FAISS
from .view_embeddings import embed_viewer

class AddDocumentEmbeddings:
    """
    A class to add new document embeddings to an existing vector store.

    Attributes:
        faiss_persist_directory (str): Directory where the FAISS index is persisted.
        embeddings (list): List of embeddings to be added to the vector store.

    Methods:
        add_to_existing_vector_store(): Adds new embeddings to the existing vector store and returns the updated store.
    """

    def __init__(self, faiss_persist_directory, embeddings):
        """
        Constructs all the necessary attributes for the AddDocumentEmbeddings object.

        Parameters:
            faiss_persist_directory (str): Directory where the FAISS index is persisted.
            embeddings (list): List of embeddings to be added to the vector store.
        """
        self.faiss_persist_directory = faiss_persist_directory
        self.embeddings = embeddings

    
    def add_to_existing_vector_store(self):
        """
        Adds new embeddings to the existing vector store and returns the updated store.

        Returns:
            The updated vector store with new embeddings added.
        """
        new_chunks = doc_processor.process_documents()
        
        # Create a FAISS extension from new documents
        self.extension = FAISS.from_documents(new_chunks, self.embeddings)
        
        # Load existing embeddings
        self.new_db = embed_viewer.load_embeddings()
        
        # Merge new embeddings into the existing vector store
        updated_vector_store = self.new_db.merge_from(self.extension)
        
        # Optionally, save the updated vector store locally
        # updated_vector_store.save_local(self.faiss_persist_directory)
        
        return updated_vector_store

# Instantiate the AddDocumentEmbeddings with configuration settings    
new_docs_embedder = AddDocumentEmbeddings(cfg.faiss_persist_directory, cfg.embeddings)