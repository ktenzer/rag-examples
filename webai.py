import os, pathlib, sys, textwrap
from typing import List
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from openai import OpenAI

# Load Environment
load_dotenv(override=True)
base_url = os.getenv("OPENAI_BASE_URL")
api_key  = os.getenv("OPENAI_API_KEY")

if not base_url or not api_key:
    sys.exit(f"ERROR: OPENAI_BASE_URL or OPENAI_API_KEY missing after load_dotenv\n"
             f"base_url={base_url!r}, api_key={api_key!r}")

print(f"Using WebAI endpoint {base_url} with key {api_key[:6]}…")

oa = OpenAI(base_url=base_url, api_key=api_key)

# Config
DOCS_DIR   = pathlib.Path("./docs").resolve()
CHROMA_DIR = pathlib.Path("./chroma_db/webai")

EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"    # ≈150 MB
CHAT_MODEL     = "webai"            # whatever WebAI exposes

K_RETRIEVE    = 4
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

# Embedding wrapper for Chroma
class HFEmbedding(EmbeddingFunction):
    """CPU‑only embedding using sentence‑transformers."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")
        self._dims = self.model.get_sentence_embedding_dimension()

    def name(self): return f"hf-{EMB_MODEL_NAME}"
    def dimensions(self): return self._dims

    def __call__(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

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
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def build_vector_store(chunks):
    ef = HFEmbedding(EMB_MODEL_NAME)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False, is_persistent=True)
    )

    name = "rag_docs"
    if name in [c.name for c in client.list_collections()]:
        col = client.get_collection(name, embedding_function=ef)
    else:
        col = client.create_collection(name, embedding_function=ef)
        col.add(
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[f"chunk-{i}" for i in range(len(chunks))],
        )
    return col

def shorten(txt, width=160):
    return textwrap.shorten(" ".join(txt.split()), width=width)

def label(meta: dict, idx: int):
    fn = os.path.basename(meta.get("source", "unknown"))
    page = meta.get("page"); page_str = f" p{page+1}" if isinstance(page, int) else ""
    return f"Source {idx}{':' if fn else ''} {fn}{page_str}"

# Main
def main():
    if not DOCS_DIR.exists():
        sys.exit(f"Docs folder {DOCS_DIR} not found.")

    print("Loading & splitting documents …")
    chunks = load_documents(); print(f"→ {len(chunks)} chunks")

    print("Building / opening Chroma vector store …")
    store = build_vector_store(chunks)

    print("Ready! (chat via WebAI, embeddings local)  ctrl‑c to quit\n")
    while True:
        try:
            user_q = input("User: ").strip()
            if not user_q: continue

            # Retrieve
            r = store.query(query_texts=[user_q], n_results=K_RETRIEVE)
            docs, metas = r["documents"][0], r["metadatas"][0]

            # Show what was retrieved (debug)
            print("\n Retrieved chunks:")
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                print(f"{i:>2}. {label(m, i)} — {shorten(d, 200)}")
            print("")

            # Build prompt with *complete* chunk text
            source_blocks = [
                f"Source {i} ({label(m, i)}):\n{d}"
                for i, (d, m) in enumerate(zip(docs, metas), 1)
            ]
            sources_text = "\n\n".join(source_blocks)

            prompt = (
                "You are a helpful assistant. Use ONLY the sources below.\n\n"
                f"User question: {user_q}\n\n"
                f"{sources_text}\n\n"
                "When you cite a fact, reference it like (Source 2)."
            )

            resp = oa.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            print(f"\nAssistant: {resp.choices[0].message.content}\n")

        except KeyboardInterrupt:
            print("\nGood Bye!"); break

if __name__ == "__main__":
    main()