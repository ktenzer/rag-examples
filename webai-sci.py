import os, sys, time, textwrap, mimetypes, warnings, logging, csv, io, re, json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
)
import torch, chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from PIL import Image
from rank_bm25 import BM25Okapi
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Env
load_dotenv(override=True)
BASE_URL, API_KEY = os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY")
if not BASE_URL or not API_KEY:
    sys.exit("ERROR: OPENAI_BASE_URL / OPENAI_API_KEY missing in .env")
from openai import OpenAI
oa = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Ignore warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Models
TXT_EMBED_MODEL  = "text-embedding-3-large"
CLIP_MODEL       = "openai/clip-vit-base-patch32"
BLIP_MODEL       = "Salesforce/blip-image-captioning-base"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHARTQA_MODEL    = "google/pix2struct-chartqa-base"

print("Loading models …")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=False)

blip_proc  = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)

p2s_proc   = Pix2StructProcessor.from_pretrained(CHARTQA_MODEL)
p2s_model  = Pix2StructForConditionalGeneration.from_pretrained(CHARTQA_MODEL)

reranker   = CrossEncoder(RERANK_MODEL)

ocr_model  = ocr_predictor(
                pretrained=True,
                det_arch="db_resnet50",
                reco_arch="crnn_vgg16_bn",
                assume_straight_pages=True
            )
print("Models ready\n")

# Embeddings
class SciEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        self.dim   = self.model.get_sentence_embedding_dimension()

    def name(self): return "bge-large-en-v1.5"
    def dimensions(self): return self.dim

    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

class CLIPTextEF(EmbeddingFunction):
    def __init__(self): self.proc, self.model = clip_proc, clip_model
    def name(self): return f"clip-text-{CLIP_MODEL}"
    def dimensions(self): return 512
    def __call__(self, texts):
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True)
            return self.model.get_text_features(**inp).cpu().numpy().tolist()

clip_text_ef = CLIPTextEF()

def clip_embed_image(path: str):
    with torch.no_grad():
        x = clip_proc(images=Image.open(path), return_tensors="pt")
        return clip_model.get_image_features(**x)[0].cpu().numpy().tolist()

# OCR / Caption
def blip_caption(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        out = blip_model.generate(**blip_proc(images=img, return_tensors="pt"), max_new_tokens=40)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def chartqa_caption(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    inputs = p2s_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_new_tokens=64)
    return p2s_proc.decode(out[0], skip_special_tokens=True)

def doctr_ocr(path: Path) -> str:
    doc = DocumentFile.from_images(str(path))
    out = ocr_model(doc).export()
    return " ".join(
        w["value"]
        for p in out["pages"]
        for b in p["blocks"]
        for l in b["lines"]
        for w in l["words"]
    )

# Text splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

def md_split(docs, chunk=800, overlap=80):
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    out: List[Document] = []
    print("Chunking text …")
    t0 = time.time()
    for doc in docs:
        for sec in header.split_text(doc.page_content):
            for ch in rc.split_text(sec.page_content):
                out.append(Document(page_content=ch, metadata=doc.metadata))
    print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
    return out

# PDF and Table loaders
from unstructured.partition.auto import partition
import pdfplumber

DOCS_DIR   = Path("./docs").resolve()
CHROMA_DIR = Path("./chroma_db/webai")

def ocr_pdf(path: Path):
    print(f"OCR {path.name}")
    elems = partition(filename=str(path), strategy="hi_res", languages=["eng"])
    return "\n".join(e.text for e in elems if e.text)

def csv_to_sentences(raw_csv: str, hdr: List[str]) -> List[str]:
    out = []
    for row in csv.reader(io.StringIO(raw_csv)):
        if row == hdr: continue
        out.append("Row -> " + ", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(hdr, row)))
    return out

def tables_docs(path: Path) -> List[Document]:
    docs = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                hdr, *rows = tbl
                hdr = ["" if c is None else str(c) for c in hdr]

                # Markdown table
                md = "| " + " | ".join(hdr) + " |\n" + "|---" * len(hdr) + "|\n"
                for r in rows:
                    md += "| " + " | ".join("" if c is None else str(c) for c in r) + " |\n"
                docs.append(Document(page_content=md, metadata={"source": str(path), "table_md": True}))

                # JSON table
                docs.append(Document(page_content=json.dumps({"headers": hdr, "rows": rows}, ensure_ascii=False),
                                     metadata={"source": str(path), "table_json": True}))

                # Row sentences
                raw_csv = "\n".join([",".join(hdr)] + [",".join("" if c is None else str(c) for c in r) for r in rows])
                for sent in csv_to_sentences(raw_csv, hdr):
                    docs.append(Document(page_content=sent, metadata={"source": str(path), "table_row": True}))
    return docs

# Loader
def load_docs():
    text, images = [], []
    for fp in DOCS_DIR.rglob("*"):
        if fp.is_dir(): continue
        mime = mimetypes.guess_type(fp)[0] or ""

        if mime.startswith("image"):
            images.append(Document(page_content="", metadata={"image_path": str(fp)}))

            cap   = blip_caption(fp)
            chart = chartqa_caption(fp)
            ocr   = doctr_ocr(fp)

            combo = " • ".join(t for t in (cap, chart, ocr) if t)
            text.append(Document(page_content=combo,
                                 metadata={"image_path": str(fp),
                                           "caption": True,
                                           "chartqa": bool(chart),
                                           "ocr": bool(ocr)}))

        elif fp.suffix.lower() == ".pdf":
            text.append(Document(page_content=ocr_pdf(fp), metadata={"source": str(fp)}))
            text.extend(tables_docs(fp))

        elif fp.suffix.lower() in {".txt", ".md"}:
            print(f"Load {fp.name}")
            text.append(Document(page_content=fp.read_text(), metadata={"source": str(fp)}))

    print(f"Loaded {len(text)} text docs & {len(images)} images\n")
    return text, images, images 

# Chroma DB Helpers
def open_if_exists(client):
    names = {c.name for c in client.list_collections()}
    return (
        client.get_collection("text",   embedding_function=SciEmbedding()) if "text"   in names else None,
        client.get_collection("images", embedding_function=clip_text_ef)  if "images" in names else None,
    )

def build_stores(txt_chunks, img_docs, client):
    print("Embedding & writing to Chroma …")
    t0 = time.time()
    txt_col = client.get_or_create_collection("text", embedding_function=SciEmbedding())
    txt_col.add(
        documents=[d.page_content for d in txt_chunks],
        metadatas=[d.metadata for d in txt_chunks],
        ids=[f"t{idx}" for idx in range(len(txt_chunks))]
    )
    img_col = client.get_or_create_collection("images", embedding_function=clip_text_ef)
    if img_docs:
        vecs = [clip_embed_image(d.metadata["image_path"]) for d in img_docs]
        img_col.add(
            documents=[""] * len(img_docs),
            metadatas=[d.metadata for d in img_docs],
            embeddings=vecs,
            ids=[f"i{idx}" for idx in range(len(img_docs))]
        )
    print(f"Vector DB ready ({time.time()-t0:.1f}s)\n")
    return txt_col, img_col

# BM25 Wrapper
class BM25Store:
    def __init__(self, docs_lower: List[str]):
        self.model = BM25Okapi([d.split() for d in docs_lower])
    def query(self, q: str, k: int):
        idxs = self.model.get_top_n(q.split(), range(len(self.model.doc_freqs)), n=k)
        return idxs

# Utils
def shorten(t, w=200): return textwrap.shorten(" ".join(t.split()), width=w)
def label(meta, idx):
    if meta.get("image_path") and meta.get("caption"):
        src = Path(meta["image_path"]).name + " (image)"
    else:
        src = Path(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr") if meta.get(k)]
    return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

def rerank(query: str, docs: List[str], metas: List[dict], keep=4):
    scores = reranker.predict([(query, d if d else " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

# Main
def main():
    # Persistent Chroma client
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Text collection
    try:
        txt_col = client.get_collection("text", embedding_function=SciEmbedding())
    except Exception:
        txt_col = None

    # If text vectors exist, we assume the DB is already built
    if txt_col and txt_col.count() > 0:
        print(f"Found existing Chroma DB ({txt_col.count()} text vectors) – skipping ingest\n")

        try:
            img_col = client.get_collection("images", embedding_function=clip_text_ef)
        except Exception:
            img_col = client.get_or_create_collection("images", embedding_function=clip_text_ef)

    else:
        # No usable vectors, build everything from scratch
        print("No text vectors detected running full ingest (OCR, chunk, embed) …")
        text_docs, image_docs, img_docs = load_docs()
        chunks   = md_split(text_docs)
        txt_col, img_col = build_stores(chunks, img_docs, client)

    # Build BM25
    corpus_docs  = txt_col.get()["documents"]
    corpus_metas = txt_col.get()["metadatas"]
    bm25 = BM25Store([d.lower() for d in corpus_docs])

    print("Ready: ask me anything (Ctrl‑C to quit)\n")

    # Chat Loop
    while True:
        try:
            q = input("User: ").strip()
            if not q:
                continue

            # Dense retrieval
            txt = txt_col.query(query_texts=[q], n_results=20)
            docs_vec, metas_vec = txt["documents"][0], txt["metadatas"][0]

            # BM25 lexical retrieval
            idxs = bm25.query(q.lower(), k=10)
            docs_bm  = [corpus_docs[i]  for i in idxs]
            metas_bm = [corpus_metas[i] for i in idxs]

            # Merge & cross‑encoder re‑rank
            docs_all  = docs_vec + docs_bm
            metas_all = metas_vec + metas_bm
            docs, metas = rerank(q, docs_all, metas_all, keep=4)

            # CLIP image retrieval
            if img_col.count() > 0:
                clip_vec = clip_text_ef([q])[0]
                img = img_col.query(query_embeddings=[clip_vec], n_results=2)
                docs  += img["documents"][0]
                metas += img["metadatas"][0]

            # Display context
            print("\n🔎 Retrieved context:")
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                snippet = "<image>" if m.get("image_path") else shorten(d)
                print(f"  {i}. {label(m, i)} — {snippet}")
            print("")

            # Build LLM prompt
            blocks = []
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                content = f"<image at {m['image_path']}>" if m.get("image_path") else d
                blocks.append(f"{label(m, i)}\n{content}")

            prompt = (
                "You are a helpful assistant. Use ONLY the sources below.\n\n"
                f"User question: {q}\n\n" + "\n\n".join(blocks) +
                "\n\nCite facts like (Source 2)."
            )

            ans = oa.chat.completions.create(
                model="webai",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            ).choices[0].message.content
            print("\nAssistant:", ans, "\n")

        except KeyboardInterrupt:
            print("\nGood Bye")
            break

if __name__ == "__main__":
    main()