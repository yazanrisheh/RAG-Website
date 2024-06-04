from langchain.text_splitter import RecursiveCharacterTextSplitter
from doc_process.document_loaders import DocumentLoader


class DocumentProcessor:
    """
    A class to process documents within a specified directory.

    This class is responsible for loading documents from a directory,
    splitting them into chunks of a specified size, and potentially
    overlapping chunks as per the given parameters.

    Attributes:
        directory_path (str): The path to the directory containing documents.
        chunk_size (int): The size of each chunk after splitting the documents.
        chunk_overlap (int): The number of characters that chunks will overlap.
    """

    def __init__(self, chunk_size=3000, chunk_overlap=500):
        """
        Initializes the DocumentProcessor with a directory path, chunk size, and chunk overlap.

        :param directory_path: The path to the directory containing documents.
        :param chunk_size: The size of each chunk after splitting the documents. Defaults to 3000.
        :param chunk_overlap: The number of characters that chunks will overlap. Defaults to 500.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def text_splitter_by_char(self):
        """
        Creates a RecursiveCharacterTextSplitter instance for splitting documents.

        This method logs the splitting process and returns a configured instance
        of RecursiveCharacterTextSplitter based on the chunk size and overlap.

        :return: An instance of RecursiveCharacterTextSplitter.
        """

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def process_documents(self):
        """
        Processes all documents by loading and then splitting them into chunks.

        This method loads documents using `load_documents` and then splits them
        using the `text_splitter_by_char` method.

        :return: A list of document chunks.
        """
        doc_loader = DocumentLoader()
        new_docs = doc_loader.load_documents()
        splitter = self.text_splitter_by_char()
        chunks = splitter.split_documents(new_docs)

        return chunks


# doc_processor = DocumentProcessor()
