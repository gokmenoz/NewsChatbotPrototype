# utils.py

import os
import json
import requests
import time
import random
import boto3
import botocore.exceptions
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Claude / Bedrock setup
session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


def fetch_from_newsapi(query: str) -> List[Dict]:
    if not NEWSAPI_KEY:
        print("⚠️ Missing NEWSAPI_KEY")
        return []

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": 10,
                "apiKey": NEWSAPI_KEY
            },
            timeout=10
        )
        articles = response.json().get("articles", [])
        return [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"]
            } for a in articles
        ]
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return []


def fetch_from_gnews(query: str) -> List[Dict]:
    if not GNEWS_API_KEY:
        print("⚠️ Missing GNEWS_API_KEY")
        return []

    try:
        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "lang": "en",
                "max": 10,
                "token": GNEWS_API_KEY
            },
            timeout=10
        )
        articles = response.json().get("articles", [])
        return [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"]
            } for a in articles
        ]
    except Exception as e:
        print(f"❌ GNews error: {e}")
        return []


def fetch_from_newsdata(query: str) -> List[Dict]:
    if not NEWSDATA_API_KEY:
        print("⚠️ Missing NEWSDATA_API_KEY")
        return []

    try:
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params={
                "q": query,
                "language": "en",
                "apikey": NEWSDATA_API_KEY
            },
            timeout=10
        )
        articles = response.json().get("results", [])
        return [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["link"],
                "source": a["source_id"],
                "publishedAt": a["pubDate"]
            } for a in articles
        ]
    except Exception as e:
        print(f"❌ NewsData.io error: {e}")
        return []


def fetch_news_articles(query: str) -> List[Dict]:
    """Fetch news from all three sources and combine."""
    newsapi = fetch_from_newsapi(query)
    gnews = fetch_from_gnews(query)
    newsdata = fetch_from_newsdata(query)
    return newsapi + gnews + newsdata


def get_top_articles(query: str, articles: List[Dict], top_k=5) -> List[str]:
    """Return top-k semantically relevant article descriptions to the query."""
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    article_texts = [
        (a.get("title") or "") + " " + (a.get("description") or "")
        for a in articles
    ]
    article_embeddings = embedder.encode(article_texts, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, article_embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        idx = hit["corpus_id"]
        article = articles[idx]
        title = article.get("title", "Untitled")
        source = article.get("source", "Unknown Source")
        published = article.get("publishedAt", "Unknown Date")
        description = article.get("description", "(No description)")
        url = article.get("url", "")

        snippet = f"{title} ({source}, {published}):\n{description}\n{url}"
        results.append(snippet)

    return results


def retrieve(query: str, articles: List[Dict], embedder: SentenceTransformer, top_k: int = 5) -> List[str]:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    article_texts = [
        (a.get("title") or "") + " " + (a.get("description") or "")
        for a in articles
    ]
    article_embeddings = embedder.encode(article_texts, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, article_embeddings, top_k=top_k)[0]

    return [article_texts[hit["corpus_id"]] for hit in hits]


def build_rag_prompt(context: str, question: str) -> str:
    return f"You are a news assistant. Use the following articles to answer:\n\n{context}\n\nQuestion: {question}"


def call_claude_stream(prompt=None, messages_override=None, retries=5, base_delay=2):
    messages = messages_override or [{"role": "user", "content": prompt}]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    for attempt in range(retries):
        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            def stream_generator():
                for event in response["body"]:
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"])
                        if chunk_data.get("type") == "content_block_delta":
                            delta = chunk_data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text

            return stream_generator()

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"⏳ Throttled. Retrying in {wait:.2f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            time.sleep(1)

    raise RuntimeError("❌ Claude streaming call failed after retries")
