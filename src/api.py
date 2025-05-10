from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.utils import (
    fetch_news_articles,
    call_claude_stream,
    build_rag_prompt,
    retrieve
)

app = FastAPI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []


@app.get("/")
def root():
    return {"message": "✅ News Chatbot API is running."}


@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query

    # Step 1: Fetch articles from real news APIs
    articles = fetch_news_articles(query)

    if not articles:
        # Fallback to Claude if no articles found
        fallback_prompt = f"You are a helpful news assistant. Answer this:\n\n{query}"
        stream = call_claude_stream(fallback_prompt)
        return StreamingResponse(content=stream, media_type="text/plain")

    # Step 2: Embed and retrieve top relevant articles
    docs = retrieve(query, articles, embedder, top_k=5)

    # Step 3: Construct prompt and stream Claude response
    context = "\n---\n".join(docs)
    rag_prompt = build_rag_prompt(context, query)
    stream = call_claude_stream(rag_prompt)

    return StreamingResponse(content=stream, media_type="text/plain")