from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import Counter
import io
import os
import re
import json
import requests
from typing import Optional

app = FastAPI(title="echo-nemo-1.0 RAG service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# persistent Chroma client — data survives restarts on Render disk
chroma_client = chromadb.PersistentClient(
    path="/data/chroma",
    settings=Settings(anonymized_telemetry=False),
)

# load embedding model once at startup (cached after first load)
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model ready.")


# ── helpers ──────────────────────────────────────────────────────────────────

def get_collection(user_id: str):
    """Get or create a per-user Chroma collection."""
    name = f"user_{re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)}"
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def text_to_markdown(text: str, source: str) -> str:
    """Clean raw text into markdown."""
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return f"# {source}\n\n" + "\n\n".join(lines)


def extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


def extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_csv(data: bytes) -> str:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data))
    return df.to_markdown(index=False)


def extract_url(url: str) -> str:
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2"}


class IngestURLRequest(BaseModel):
    user_id: str
    url: str


@app.post("/ingest/url")
def ingest_url(req: IngestURLRequest):
    try:
        raw = extract_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    md = text_to_markdown(raw, req.url)
    chunks = chunk_text(md)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from URL")
    

    # in ingest_url, before the upsert:
    existing = collection.get(where={"source": req.url}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    collection = get_collection(req.user_id)

    ids = [f"{req.url}_{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": req.url, "type": "url", "chunk": i} for i in range(len(chunks))],
    )

    return {"ingested": len(chunks), "source": req.url}


@app.post("/ingest/file")
async def ingest_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    data = await file.read()
    filename = file.filename or "unknown"
    ext = filename.lower().rsplit(".", 1)[-1]

    try:
        if ext == "pdf":
            raw = extract_pdf(data)
        elif ext in ("docx", "doc"):
            raw = extract_docx(data)
        elif ext == "csv":
            raw = extract_csv(data)
        elif ext in ("txt", "md"):
            raw = data.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    md = text_to_markdown(raw, filename)
    chunks = chunk_text(md)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted")

    collection = get_collection(user_id)

    # remove any existing chunks for this exact filename before re-ingesting
    # prevents duplicate/stale chunks when the same file is uploaded twice
    existing = collection.get(where={"source": filename}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": filename, "type": ext, "chunk": i} for i in range(len(chunks))],
    )

    return {"ingested": len(chunks), "source": filename}


class IngestTextRequest(BaseModel):
    user_id: str
    content: str
    source: str
    source_type: str = "transcript"


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    """Ingest plain text — used for Echo video transcripts."""
    md = text_to_markdown(req.content, req.source)
    chunks = chunk_text(md)

    if not chunks:
        raise HTTPException(status_code=400, detail="Empty content")
    
    # in ingest_text, before the upsert:
    existing = collection.get(where={"source": req.source}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    collection = get_collection(req.user_id)

    ids = [f"{req.source}_{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{
            "source": req.source,
            "type": req.source_type,
            "chunk": i,
        } for i in range(len(chunks))],
    )

    return {"ingested": len(chunks), "source": req.source}


class QueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 5


@app.post("/query")
def query(req: QueryRequest):
    collection = get_collection(req.user_id)

    count = collection.count()
    if count == 0:
        return {
            "chunks": [],
            "sources": [],
            "has_context": False,
        }

    q_embedding = embedder.encode([req.question], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(req.top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = list({m["source"] for m in metadatas})

    return {
        "chunks": chunks,
        "sources": sources,
        "distances": distances,
        "has_context": len(chunks) > 0,
    }


class ScoredQueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 8
    boost_sources: list[dict] = []


@app.post("/query/scored")
def query_scored(req: ScoredQueryRequest):
    """Retrieve chunks then apply score boosts from feedback history."""
    collection = get_collection(req.user_id)
    count = collection.count()

    if count == 0:
        return {"chunks": [], "sources": [], "has_context": False, "chunk_ids": []}

    q_embedding = embedder.encode(
        [req.question], show_progress_bar=False
    ).tolist()

    # ids are always returned by chroma — do NOT put "ids" in include
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(req.top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]  # always present regardless of include

    # build boost map from feedback history
    boost_map: dict[str, float] = {
        b["source"]: b["score"] for b in req.boost_sources
    }

    # score = (1 - cosine_distance) + feedback_boost
    scored = []
    for i, chunk in enumerate(chunks):
        source = metadatas[i].get("source", "")
        base_score = 1 - distances[i]
        boost = boost_map.get(source, 0.0)
        final_score = base_score + (boost * 0.15)  # boost up to 15% weight
        scored.append({
            "chunk": chunk,
            "source": source,
            "score": final_score,
            "id": ids[i],
            "metadata": metadatas[i],
        })

    # re-rank by final score
    scored.sort(key=lambda x: x["score"], reverse=True)

    # return top 5 after re-ranking
    top = scored[:5]

    return {
        "chunks": [s["chunk"] for s in top],
        "sources": list({s["source"] for s in top}),
        "chunk_ids": [s["id"] for s in top],
        "scores": [s["score"] for s in top],
        "has_context": len(top) > 0,
    }


class QueryExpansionRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/expand-query")
def expand_query(req: QueryExpansionRequest):
    """
    Simple query expansion — extracts key terms from conversation history
    to make the current question more specific.
    Uses no LLM, just NLP heuristics — keeps it free and fast.
    """
    combined = req.question
    if req.history:
        # extract nouns/key phrases from last 2 exchanges
        recent = " ".join(
            m["content"] for m in req.history[-4:] if m.get("role") == "user"
        )
        # add unique words from recent context that aren't in the current query
        current_words = set(req.question.lower().split())
        extra_words = [
            w for w in recent.split()
            if len(w) > 4 and w.lower() not in current_words
            and re.match(r"^[a-zA-Z]+$", w)
        ]
        # take top 3 most frequent extra context words
        top_extra = [w for w, _ in Counter(extra_words).most_common(3)]
        if top_extra:
            combined = req.question + " " + " ".join(top_extra)

    return {"expanded_question": combined}


class DocumentListRequest(BaseModel):
    user_id: str


@app.post("/documents")
def list_documents(req: DocumentListRequest):
    """List all unique sources ingested by a user."""
    collection = get_collection(req.user_id)
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}

    results = collection.get(include=["metadatas"])
    sources: dict[str, dict] = {}
    for m in results["metadatas"]:
        src = m["source"]
        if src not in sources:
            sources[src] = {"source": src, "type": m.get("type", "unknown"), "chunks": 0}
        sources[src]["chunks"] += 1

    return {
        "documents": list(sources.values()),
        "total_chunks": collection.count(),
    }


class DeleteDocumentRequest(BaseModel):
    user_id: str
    source: str


@app.post("/documents/delete")
def delete_document(req: DeleteDocumentRequest):
    """Delete all chunks for a specific source document."""
    collection = get_collection(req.user_id)

    results = collection.get(
        where={"source": req.source},
        include=["metadatas"],
    )
    ids = results["ids"]

    print(f"Deleting {len(ids)} chunks for source '{req.source}' (user: {req.user_id})")

    if ids:
        collection.delete(ids=ids)

    # verify deletion actually happened
    verify = collection.get(where={"source": req.source}, include=[])
    remaining = len(verify["ids"])

    if remaining > 0:
        print(f"WARNING: {remaining} chunks still remain after delete attempt")

    return {
        "deleted": len(ids),
        "source": req.source,
        "remaining": remaining,
    }


@app.delete("/user/{user_id}")
def delete_user_data(user_id: str):
    """Delete all data for a user — GDPR compliance."""
    name = f"user_{re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)}"
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass
    return {"deleted": True, "user_id": user_id}