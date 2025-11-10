from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx

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
    thriftbooks_buyback: bool = False
    thriftbooks_price: Optional[float] = None
    thriftbooks_raw_status: Optional[str] = None
    source: Optional[str] = None


# ============================================================
# External helpers
# ============================================================

OPENLIBRARY_URL = "https://openlibrary.org/api/books"


async def lookup_openlibrary(isbn: str) -> (Optional[str], Optional[str]):
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


async def check_thriftbooks_buyback(isbn: str) -> (bool, Optional[float], str):
    """
    Placeholder for ThriftBooks buyback check.

    Currently does NOT call ThriftBooks. It returns:
      (False, None, "not_implemented")

    Later you can:
      - Inspect ThriftBooks' network requests in your browser,
      - Replicate them here with httpx for your personal use.
    """
    return False, None, "not_implemented"


# ============================================================
# Main endpoint
# ============================================================

@app.post("/check-isbns", response_model=List[ISBNResult])
async def check_isbns(payload: ISBNRequest):
    """
    Accepts a list of ISBNs, normalizes & de-duplicates them,
    checks (stub) ThriftBooks, looks up metadata via Open Library,
    and returns structured results.
    """
    results: List[ISBNResult] = []

    seen = set()
    cleaned_isbns: List[str] = []

    # Normalize input
    for raw in payload.isbns:
        if not raw:
            continue
        isbn = (
            raw.strip()
            .replace("-", "")
            .replace(" ", "")
        )
        if not isbn:
            continue
        if isbn in seen:
            continue
        seen.add(isbn)
        cleaned_isbns.append(isbn)

    # Process each ISBN
    for isbn in cleaned_isbns:
        tb_ok, tb_price, tb_status = await check_thriftbooks_buyback(isbn)
        title, author = await lookup_openlibrary(isbn)

        if tb_ok:
            if title or author:
                source = "thriftbooks+openlibrary"
            else:
                source = "thriftbooks_only"
        else:
            source = "openlibrary_only" if (title or author) else "unknown"

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
# Healthcheck
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "bookshelf-buyback-api is running",
        "endpoints": ["/check-isbns"],
    }
