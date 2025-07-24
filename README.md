# rag-examples
Local RAG pipeline that embeds a folder of documents, stores them in Chroma, and chats via ChatGPT or webAI

### How it works?
1. Model Setup:
Loads all required AI models (OCR, captioning, embedding, reranking, etc.) and configures API keys.

2. Embedding Functions:
Defines functions to turn text and images into embeddings, allowing semantic search in the vector database.

3. Document Processing:
Converts source PDFs, images, and text files into small “chunks”—including OCR text, image captions, table extractions, and more.

4. Document Ingestion:
Scans your documents folder, applies all preprocessing, and creates structured Document objects for everything.

5. Database Building:
Embeds all chunks and stores them in ChromaDB (vector database), creating collections for both text and images.

6. Retrieval Pipeline:
When a user asks a question, retrieves the most relevant chunks using semantic (vector), keyword (BM25), MMR diversity, and reranking. Also finds relevant images using CLIP.

7. Prompt Building & LLM Answer:
Builds a prompt with the user’s question and retrieved context, then sends it to the Llama 3B model to generate an answer, requiring citations from sources.

8. Chat Loop:
Repeats the retrieval and answer steps for every new user question, allowing interactive, document-grounded Q&A.

## Add Documents
Add pdf files under directory called docs

## Install Dependencies
```bash
$ poetry install
```

## Configuration
```bash
$ cp .env.example .env
```

Setup environment depending on using openai or webAI

## OpenAI (text RAG)
```bash
$ poetry run python chatgpt.py
```

## webAI (text RAG)
Ensure API and LLM Chat elements are running on correct port and with the right api key

```bash
$ poetry run python webai.py
```

## webAI (Scientific RAG)
Ensure API and LLM Chat elements are running on correct port and with the right api key

```bash
$ poetry run python webai-sci.py
```



