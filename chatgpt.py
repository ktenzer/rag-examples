import os, pathlib, sys, textwrap
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Config
DOCS_DIR      = pathlib.Path("./docs").resolve()
CHROMA_DIR    = pathlib.Path("./chroma_db/openai")
EMBED_MODEL   = "text-embedding-3-large"
K_RETRIEVE    = 4
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

# Helpers
def load_documents() -> List[str]:
    loaders = [
        DirectoryLoader(str(DOCS_DIR), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(DOCS_DIR), glob="**/*.md",  loader_cls=TextLoader),
        DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    docs = []
    for l in loaders:
        docs.extend(l.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

def build_vector_store(chunks):
    embedding_fn = OpenAIEmbeddingFunction(
        model_name=EMBED_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    name = "rag_docs"
    if name in [c.name for c in client.list_collections()]:
        col = client.get_collection(name, embedding_function=embedding_fn)
    else:
        col = client.create_collection(name, embedding_function=embedding_fn)
        col.add(
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[f"chunk-{i}" for i in range(len(chunks))],
        )
    return col

def shorten(txt: str, width: int = 160) -> str:
    return textwrap.shorten(" ".join(txt.split()), width=width)

def label(meta: dict, idx: int) -> str:
    """Return a stable, human label → 'Source 1: foo.pdf p3'."""
    fn = os.path.basename(meta.get("source", "unknown"))
    page = meta.get("page")
    if isinstance(page, int):
        page += 1  # PyPDFLoader pages are 0‑based
    page_part = f" p{page}" if page is not None else ""
    return f"Source {idx}{':' if fn else ''} {fn}{page_part}"

# Main
def main():
    load_dotenv()
    if not DOCS_DIR.exists():
        sys.exit(f"ERROR: Docs folder {DOCS_DIR} not found.")

    oa = OpenAI()

    print("Loading & splitting documents …")
    chunks = load_documents()
    print(f"   → {len(chunks)} chunks")

    print("Building / opening Chroma vector store …")
    store = build_vector_store(chunks)

    print("Ready!  Ask me anything (ctrl‑c to quit)\n")
    while True:
        try:
            user_q = input("User: ").strip()
            if not user_q:
                continue

            # Retrieve
            r = store.query(query_texts=[user_q], n_results=K_RETRIEVE)
            docs      = r["documents"][0]
            metas     = r["metadatas"][0]

            # DEBUG: retrieved chunks
            print("\nRetrieved chunks:")
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                print(f"  {i:>2}. {label(meta, i)} — {shorten(doc, 200)}")
            print("")

            # Build sources section
            source_lines = []
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                source_lines.append(f"- {label(meta, i)} — {shorten(doc)}")
            sources_block = "\n".join(source_lines)

            # System Prompt
            prompt = (
                "You are a helpful assistant. Use ONLY the sources below to answer.\n\n"
                f"User question: {user_q}\n\n"
                "Sources:\n"
                f"{sources_block}\n\n"
                "If you use a fact from a source, cite it like (Source 2)."
            )

            # DEBUG: full prompt
            print("Prompt to OpenAI:\n" + "-"*60 + f"\n{prompt}\n" + "-"*60 + "\n")

            # Chat response
            resp = oa.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            print(f"ChatGPT: {resp.choices[0].message.content}\n")

        except KeyboardInterrupt:
            print("\nGood Bye!")
            break

if __name__ == "__main__":
    main()