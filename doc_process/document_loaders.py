from langchain_community.document_loaders import WebBaseLoader
import bs4

class DocumentLoader:
    def __init__(self):
        self.urls = (
        "https://schemdraw.readthedocs.io/en/stable/",
        "https://schemdraw.readthedocs.io/en/stable/usage/start.html"
        )

    def load_documents(self):
        docs = [
            WebBaseLoader(
                url, bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("body")))
            ).load()
            for url in self.urls
        ]
        docs_list = [item for sublist in docs for item in sublist]

        return docs_list


# doc_loader = DocumentLoader()
