# Patch Notes — Policy RAG v1.0 → v1.1

Four files were patched. Drop them into your `app/` folder,
replacing the originals.

---

## Files changed

### `document_loader.py`
**Bug fixed:** `PyPDFLoader`, `UnstructuredHTMLLoader`, and `Docx2txtLoader`
do not accept an `encoding=` keyword argument. Passing one raises a
`TypeError` and crashes ingestion for every PDF and HTML file.

**What changed:**
- `get_loader_for_file()` now only passes `encoding='utf-8'` to `TextLoader`
  (the one loader that accepts it).
- `MarkdownLoader` import removed — it was dropped from
  `langchain-community 0.0.10`; `.md` files now load via `TextLoader`
  (Markdown is plain text, so this is lossless).
- All imports updated from `langchain.document_loaders` →
  `langchain_community.document_loaders`.

---

### `vector_store.py`
**Bug fixed:** `add_documents()` used sequential IDs (`doc_0`, `doc_1` …).
Calling it a second time (e.g. after `/reindex`) raised a ChromaDB
duplicate-ID error and left the index in a broken state.

**What changed:**
- IDs are now MD5 hashes of the first 200 chars of `page_content` + the
  `chunk_id` metadata field. Same chunk → same ID, always.
- `collection.add()` replaced with `collection.upsert()` so re-indexing
  overwrites existing chunks rather than erroring.
- `delete_collection()` now also resets `self.collection = None` so a
  subsequent `create_collection()` call works correctly.
- Imports updated to `langchain_community.embeddings`.

---

### `rag_pipeline.py`
**Bug fixed:** `langchain.chat_models` and `langchain.chains` imports are
deprecated in LangChain 0.1.x and removed in 0.2.x. On some installs
these imports resolve to stubs that raise `ImportError` at runtime.

**What changed:**
- LLM imports now try the new package-specific path first
  (`langchain_openai`, `langchain_anthropic`) with a graceful fallback to
  the old path for users still on 0.1.x.
- `chain.invoke()` is tried first (0.2.x API); falls back to `chain.run()`
  for 0.1.x.

---

### `main.py`
**Bug fixed:** `python-dotenv` was listed as a dependency but
`load_dotenv()` was never called. All `os.getenv()` calls in `config.py`
silently returned `''`, causing the RAG pipeline to raise:
    `ValueError: No LLM provider configured`
even when a valid `.env` file was present.

**What changed:**
- `from dotenv import load_dotenv; load_dotenv()` added at the very top,
  before any import that reads environment variables.
- `sys.path` hack replaced with a stable path that adds `app/` (the
  directory containing `config.py`) regardless of where the script is
  launched from.

---

## How to apply

```
project_final/
└── app/
    ├── document_loader.py   ← replace with patched version
    ├── vector_store.py      ← replace with patched version
    ├── rag_pipeline.py      ← replace with patched version
    └── main.py              ← replace with patched version
```

No changes to `config.py`, `text_splitter.py`, `evaluate.py`, or any
policy documents are required.

## Recommended: also install updated LangChain packages

```bash
pip install langchain-openai langchain-anthropic langchain-community
```

This satisfies the new import paths used in the patched files and is
forward-compatible with LangChain 0.2.x.
