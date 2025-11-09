from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from typing import List, Optional

app = FastAPI(title="Shelf Scanner Helper API")

# ---------- Models ----------

class ISBNRequest(BaseModel):
    isbns: List[str]


class ISBNResult(BaseModel):
    isbn: str
    title: Optional[str] = None
    author: Optional[str] = None
    thriftbooks_buyback: bool = False
    thriftbooks_price: Optional[float] = None
    thriftbooks_raw_status: Optional[str] = None  # e.g. "accepted", "not_accepted", "error"
    source: Optional[str] = None  # e.g. "thriftbooks+openlibrary", "openlibrary_only"


# ---------- External helpers ----------

OPENLIBRARY_URL = "https://openlibrary.org/api/books"

async def lookup_openlibrary(isbn: str) -> (Optional[str], Optional[str]):
    """Get a best-effort (title, author) from Open Library for context."""
    try:
        params = {
            "bibkeys": f"ISBN:{isbn}",
            "format": "json",
            "jscmd": "data"
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
        return title, author_names or None
    except Exception:
        return None, None


async def check_thriftbooks_buyback(isbn: str) -> (bool, Optional[float], str):
    """
    ThriftBooks buyback probe (SKELETON).

    IMPORTANT:
    - ThriftBooks does not expose a documented public API for buyback.
    - The correct way to implement this for your PERSONAL use is:
        1. Log into the ThriftBooks buyback page in your browser.
        2. Open DevTools → Network tab.
        3. Enter an ISBN in their form.
        4. See what request is sent (URL, method, payload, headers).
        5. Replicate that here with httpx (with your auth if required).
    - Be respectful of their Terms of Use and rate limits.
    - Below is ONLY a placeholder pattern, not a real endpoint.
    """

    # --- PSEUDOCODE EXAMPLE ONLY ---
    # url = "https://www.thriftbooks.com/api/buyback/quote"  # <-- this is NOT a real URL; replace from DevTools
    # payload = {"isbns": [isbn]}
    # headers = {
    #     "User-Agent": "YourShelfScanner/1.0",
    #     # Include auth cookies or token if their API requires being logged in.
    # }
    # try:
    #     async with httpx.AsyncClient(timeout=4.0) as client:
    #         r = await client.post(url, json=payload, headers=headers)
    #     r.raise_for_status()
    #     data = r.json()
    #     # Shape of `data` depends on their real response.
    #     # Example logic:
    #     offer = data.get("offers", {}).get(isbn)
    #     if not offer:
    #         return False, None, "not_accepted"
    #     price = float(offer.get("price", 0) or 0)
    #     if price > 0:
    #         return True, price, "accepted"
    #     else:
    #         return False, None, "not_accepted"
    # except Exception:
    #     return False, None, "error"

    # For now, we return a neutral "no info"
    return False, None, "not_implemented"
    # --- END PSEUDOCODE ---


# ---------- Main endpoint ----------

@app.post("/check-isbns", response_model=List[ISBNResult])
async def check_isbns(payload: ISBNRequest):
    results: List[ISBNResult] = []

    for raw in payload.isbns:
        isbn = raw.strip().replace("-", "")
        if not isbn:
            continue

        # 1) ThriftBooks buyback check (you will implement this)
        tb_ok, tb_price, tb_status = await check_thriftbooks_buyback(isbn)

        # 2) Metadata from Open Library (or other sources)
        title, author = await lookup_openlibrary(isbn)

        # 3) Build result
        if tb_ok:
            source = "thriftbooks+openlibrary"
        else:
            source = "openlibrary_only" if (title or author) else "unknown"

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
