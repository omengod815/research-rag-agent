from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
)


def chunk_pages(pages: list[dict], source: str) -> list[Document]:
    docs = [
        Document(
            page_content=p["text"],
            metadata={"source": source, "page": p["page"]},
        )
        for p in pages
    ]
    return splitter.split_documents(docs)