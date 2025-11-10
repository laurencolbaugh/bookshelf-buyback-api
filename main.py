from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
import httpx
import os

# ============================================================
# FastAPI app setup
# ============================================================

app = FastAPI(title="Shelf Scanner Helper API")

# CORS: allow browser-based tools (file://, your site, etc.) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
# ============================================================

class ISBNRequest(BaseModel):
    isbns: List[str]


class ISBNResult(BaseModel):
    isbn: str
    title: Optional[str] = None
    author: Optional[str] = None
    # Kept for compatibility with your HTML; values are stubbed.
    thriftbooks_buyback: bool = False
    thriftbooks_price: Optional[float] = None
    thriftbooks_raw_status: Optional[str] = "not_implemented"
    source: Optional[str] = None


# ============================================================
# External helpers
# ============================================================

OPENLIBRARY_URL = "https://openlibrary.org/api/books"


async def lookup_openlibrary(isbn: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort lookup of (title, author) from Open Library."""
    try:
        params = {
            "bibkeys": f"ISBN:{isbn}",
            "format": "json",
            "jscmd": "data",
        }
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(OPENLIBRARY_URL, params=params)
        r.raise_for_status()
        data = r.json()
        info = data.get(f"ISBN:{isbn}")
        if not info:
            return None, None

        title = info.get("title")
        authors = info.get("authors") or []
        author_names = ", ".join(
            a.get("name") for a in authors if a.get("name")
        )
        return title, (author_names or None)
    except Exception:
        return None, None


async def check_thriftbooks_buyback(isbn: str):
    """
    Stub for compatibility.

    We intentionally DO NOT call ThriftBooks from the server because their
    buyback API requires your authenticated browser session.
    """
    return False, None, "not_implemented"


# ============================================================
# Main API endpoint
# ============================================================

@app.post("/check-isbns", response_model=List[ISBNResult])
async def check_isbns(payload: ISBNRequest):
    """
    Accepts a list of ISBNs, normalizes & de-duplicates them,
    looks up metadata from Open Library,
    and returns structured results (plus stubbed thriftbooks_* fields).
    """
    results: List[ISBNResult] = []

    seen = set()
    cleaned_isbns: List[str] = []

    # Normalize and dedupe
    for raw in payload.isbns:
        if not raw:
            continue
        isbn = raw.strip().replace("-", "").replace(" ", "")
        if not isbn:
            continue
        if isbn in seen:
            continue
        seen.add(isbn)
        cleaned_isbns.append(isbn)

    # Process each ISBN in order
    for isbn in cleaned_isbns:
        tb_ok, tb_price, tb_status = await check_thriftbooks_buyback(isbn)
        title, author = await lookup_openlibrary(isbn)

        if title or author:
            source = "openlibrary_only"
        else:
            source = "unknown"

        results.append(
            ISBNResult(
                isbn=isbn,
                title=title,
                author=author,
                thriftbooks_buyback=tb_ok,
                thriftbooks_price=tb_price,
                thriftbooks_raw_status=tb_status,
                source=source,
            )
        )

    return results


# ============================================================
# Serve the HTML tool
# ============================================================

@app.get("/bookshelf")
async def serve_html():
    """
    Serve the bookshelf-buyback.html file so you can open it via:
      /bookshelf
    """
    html_path = os.path.join(os.path.dirname(__file__), "bookshelf-buyback.html")
    if not os.path.exists(html_path):
        # Simple message if the file is missing
        return {
            "error": "bookshelf-buyback.html not found. Make sure it is in the repo root."
        }
    return FileResponse(html_path)


# ============================================================
# Healthcheck
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "bookshelf-buyback-api is running",
        "endpoints": ["/check-isbns", "/bookshelf"],
    }
