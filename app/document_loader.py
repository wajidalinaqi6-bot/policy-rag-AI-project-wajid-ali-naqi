import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    Docx2txtLoader,
)
from langchain.schema import Document
import config

# ── FIX: MarkdownLoader was removed from langchain-community 0.0.10+
#    Use TextLoader for .md files instead (Markdown is plain text).
TEXT_EXTENSIONS = {'.txt', '.md', '.markdown'}


def get_loader_for_file(file_path: str):
    """Get the appropriate loader based on file extension.

    FIX: PyPDFLoader, UnstructuredHTMLLoader, and Docx2txtLoader do NOT
    accept an 'encoding' kwarg.  Only TextLoader does.  Passing encoding=
    to any other loader raises a TypeError and crashes the ingestion phase.
    """
    ext = Path(file_path).suffix.lower()

    # TextLoader is the only loader that accepts encoding=
    if ext in TEXT_EXTENSIONS:
        return TextLoader(file_path, encoding='utf-8')

    loaders = {
        '.pdf':  PyPDFLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm':  UnstructuredHTMLLoader,
        '.docx': Docx2txtLoader,
    }

    loader_class = loaders.get(ext, TextLoader)

    # Fallback unknown extensions to TextLoader (with encoding)
    if loader_class is TextLoader:
        return TextLoader(file_path, encoding='utf-8')

    # All other loaders: no encoding kwarg
    return loader_class(file_path)


def load_documents(directory: str = None) -> List[Document]:
    """Load all documents from the policies directory."""
    if directory is None:
        directory = str(config.DATA_DIR)

    documents = []
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return documents

    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            try:
                loader = get_loader_for_file(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = str(
                        file_path.relative_to(directory_path.parent)
                    )
                    doc.metadata['filename'] = file_path.name
                documents.extend(docs)
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

    return documents


def load_single_document(file_path: str) -> List[Document]:
    """Load a single document."""
    try:
        loader = get_loader_for_file(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
