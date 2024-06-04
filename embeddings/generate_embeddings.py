from config.config import cfg
# from doc_process.doc_splitter import doc_processor
from langchain_community.vectorstores import FAISS
from doc_process.doc_splitter import DocumentProcessor


class EmbeddingGenerator:
    """
    A class to generate embeddings for document chunks and store them in a FAISS vector store.

    Attributes:
        faiss_persist_directory (str): Directory where the FAISS index is to be persisted.
        embeddings (list): The embeddings to be used for the document chunks.
        chunks (list): The document chunks to be embedded.

    Methods:
        generate_embeddings(): Generates embeddings for the chunks and saves them to the FAISS store.
    """

    def __init__(self, faiss_persist_directory, embeddings, chunks):
        """
        Constructs all the necessary attributes for the EmbeddingGenerator object.

        Parameters:
            faiss_persist_directory (str): Directory where the FAISS index is to be persisted.
            embeddings (list): The embeddings to be used for the document chunks.
            chunks (list): The document chunks to be embedded.
        """
        self.faiss_persist_directory = faiss_persist_directory
        self.embeddings = embeddings
        self.chunks = chunks
        self.doc_processor = DocumentProcessor()

    def generate_embeddings(self):
        """
        Generates embeddings for the chunks and saves them to the FAISS store.
        """
        # Generate a FAISS database from the document chunks using the specified embeddings

        db = FAISS.from_documents(self.doc_processor.process_documents, self.embeddings)
        print(db)
        
        # Save the generated database to the local filesystem
        db.save_local(self.faiss_persist_directory)

# Process the documents to get chunks
doc_processor = DocumentProcessor()
processed_chunks = doc_processor.process_documents()

# Create an instance of EmbeddingGenerator with the processed chunks
embed_gen = EmbeddingGenerator(cfg.faiss_persist_directory, cfg.embeddings, processed_chunks)