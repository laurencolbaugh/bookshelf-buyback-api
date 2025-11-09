from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx

# ============================================================
# FastAPI app setup
# ============================================================

app = FastAPI(title="Shelf Scanner Helper API")

# Allow calls from browser-based tools (your HTML page, etc.)
# You can later restrict allow_origins to specific domains you control.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    thriftbooks_raw_status: Optional[str] = None  # e.g. "accepted", "not_accepted", "error", "not_implemented"
    source: Optional[str] = None  # e.g. "thriftbooks+openlibrary", "openlibrary_only", "unknown"


# ============================================================
# External helpers
# ============================================================

OPENLIBRARY_URL = "https://openlibrary.org/api/books"


async def lookup_openlibrary(isbn: str) -> (Optional[str], Optional[str]):
    """
    Best-effort lookup of (title, author) from Open Library.
    Safe fallback metadata source.
    """
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
        author_names = ", ".join(a.get("name") for a in authors if a.get("name"))

        return title, (author_names or None)
    except Exception:
        # On any error, just return no metadata; caller decides how to handle.
        return None, None


async def check_thriftbooks_buyback(isbn: str) -> (bool, Optional[float], str):
    """
    Placeholder for ThriftBooks buyback check.

    Right now this does NOT call ThriftBooks.
    It just returns (False, None, "not_implemented").

    Later, for your PERSONAL use, you can:
      1. Open ThriftBooks BuyBack in your browser.
      2. Use DevTools → Network to see the real request sent when you enter an ISBN.
      3. Replicate that request here with httpx (respecting their terms and limits).

    Return format:
      - thriftbooks_buyback: bool
      - thriftbooks_price: float or None
      - thriftbooks_raw_status: string label
    """
    # --- START STUB IMPLEMENTATION ---
    return False, None, "not_implemented"
    # --- END STUB IMPLEMENTATION ---


# ============================================================
# Main endpoint
# ============================================================

@app.post("/check-isbns", response_model=List[ISBNResult])
async def check_isbns(payload: ISBNRequest):
    """
    Accepts a list of ISBNs.
    For each:
      - (stub) checks ThriftBooks buyback eligibility / price.
      - looks up title/author via Open Library.
      - returns a structured result.
    """
    results: List[ISBNResult] = []

    # Normalize and de-duplicate while preserving order
    seen = set()
    cleaned_isbns: List[str] = []
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
        # Basic length filter: 10 or 13 chars typical
        if len(isbn) not in (10, 13):
            # still include; some valid ISBNs can be edge cases, but we avoid obvious junk
            pass
        if isbn in seen:
            continue
        seen.add(isbn)
        cleaned_isbns.append(isbn)

    # Process each cleaned ISBN
    for isbn in cleaned_isbns:
        # 1) ThriftBooks buyback (currently stubbed)
        tb_ok, tb_price, tb_status = await check_thriftbooks_buyback(isbn)

        # 2) Metadata from Open Library
        title, author = await lookup_openlibrary(isbn)

        # 3) Decide source label
        if tb_ok:
            source = "thriftbooks+openlibrary" if (title or author) else "thriftbooks_only"
        else:
            if title or author:
                source = "openlibrary_only"
            else:
                source = "unknown"

        # 4) Build result entry
        result = ISBNResult(
            isbn=isbn,
            title=title,
            author=author,
            thriftbooks_buyback=tb_ok,
            thriftbooks_price=tb_price,
            thriftbooks_raw_status=tb_status,
            source=source,
        )
        results.append(result)

    return results


# ============================================================
# Healthcheck (optional, handy for Render)
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "bookshelf-buyback-api is running",
        "endpoints": ["/check-isbns"],
    }
