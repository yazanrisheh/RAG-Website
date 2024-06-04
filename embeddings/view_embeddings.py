import pandas as pd
from config.config import cfg
from langchain_community.vectorstores import FAISS

class EmbeddingViewer:
    """
    A class to view embeddings stored in a FAISS vector store and export them to a Pandas DataFrame.

    Attributes:
        faiss_persist_directory (str): Directory where the FAISS index is persisted.
        embeddings (list): List of embeddings to be loaded.

    Methods:
        load_embeddings(): Loads the embeddings from the FAISS store.
        store_to_df(store): Converts the embeddings store to a Pandas DataFrame and saves it as an Excel file.
        show_vstore(): Loads the embeddings and displays the DataFrame.
    """

    def __init__(self, faiss_persist_directory, embeddings):
        """
        Constructs all the necessary attributes for the EmbeddingViewer object.

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

        new_db = FAISS.load_local(self.faiss_persist_directory, self.embeddings)
        return new_db

    def store_to_df(self, store):
        """
        Converts the embeddings store to a Pandas DataFrame and saves it as an Excel file.

        Parameters:
            store: The FAISS store containing the embeddings.

        Returns:
            DataFrame: The embeddings in a Pandas DataFrame format.
        """
        v_dict = store.docstore._dict
        data_rows = []
        for k, v in v_dict.items():
            doc_name = v.metadata['source'].split('/')[-1]
            content = v.page_content
            data_rows.append({"chunk_id": k, "document": doc_name, "content": content})
        vector_df = pd.DataFrame(data_rows)
        vector_df.to_excel("vector_df.xlsx")
        return vector_df

    def show_vstore(self):
        """
        Loads the embeddings and displays the DataFrame containing the embeddings.
        """
        store = self.load_embeddings()
        vector_df = self.store_to_df(store)
        print(vector_df)

# Create an instance of EmbeddingViewer with configuration settings
embed_viewer = EmbeddingViewer(cfg.faiss_persist_directory, cfg.embeddings)