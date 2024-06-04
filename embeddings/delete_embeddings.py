from config.config import cfg
from .load_embeddings import embed_loader
from .view_embeddings import embed_viewer

class DocumentDeleter:
    """
    A class for deleting documents from a FAISS vector store.

    Attributes:
        faiss_persist_directory (str): Directory where the FAISS index is persisted.
        embeddings (list): List of embeddings associated with the vector store.
        store (FAISS): The FAISS vector store instance.

    Methods:
        delete_document(document_path): Deletes the document and its associated chunks from the vector store.
    """

    def __init__(self, faiss_persist_directory, embeddings, store):
        """
        Constructs all the necessary attributes for the DocumentDeleter object.

        Parameters:
            faiss_persist_directory (str): Directory where the FAISS index is persisted.
            embeddings (list): List of embeddings associated with the vector store.
            store (FAISS): The FAISS vector store instance.
        """
        self.faiss_persist_directory = faiss_persist_directory
        self.embeddings = embeddings
        self.store = store

    def delete_document(self, document_path):
        """
        Deletes the document and its associated chunks from the vector store.

        Parameters:
            document_path (str): The path of the document to be deleted.
        """
        # Convert the store to a DataFrame for easier manipulation
        vector_df = embed_viewer.store_to_df(self.store)
        
        # Find chunks associated with the document to be deleted
        chunks_list = vector_df.loc[vector_df['document'] == document_path]['chunk_id'].tolist()
        
        # Delete the chunks from the store
        self.store.delete(chunks_list)
        
        # Persist the updated store to the local directory
        self.store.save_local(self.faiss_persist_directory)
        
        # Optionally, display the updated vector store
        embed_viewer.show_vstore()

# Instantiate the DocumentDeleter with configuration settings
deleter = DocumentDeleter(cfg.faiss_persist_directory, cfg.embeddings, embed_loader.load_embeddings()) 