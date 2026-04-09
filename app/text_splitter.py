from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import config

def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """Split documents into chunks."""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n## ",
            "\n\n### ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            ""
        ],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks
