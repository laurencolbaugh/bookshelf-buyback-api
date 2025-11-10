from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Any, Dict
import httpx

# ============================================================
# FastAPI app setup
# ============================================================

app = FastAPI(title="Shelf Scanner Helper API")

# CORS: allow browser-based tools (file://, your site, etc.) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # later you can restrict to your domains
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
THRIFTBOOKS_QUOTE_URL = "https://www.thriftbooks.com/tb-api/buyback/get-quotes/"


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


def _extract_tb_offer_from_item(item: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
    """
    Interpret a single ThriftBooks quote object.

    From your observation, we have:
      - isAccepted: bool
      - quotePrice: number

    Behavior:
      - If isAccepted == true and quotePrice > 0 -> accepted
      - If isAccepted == true and no/zero price -> eligible_no_price
      - If isAccepted == false -> not_accepted
    """
    is_accepted = bool(item.get("isAccepted", False))

    # Some responses may use different casing; normalize just in case
    if "isaccepted" in item:
        is_accepted = bool(item.get("isaccepted", is_accepted))

    raw_price = item.get("quotePrice", None)
    if raw_price is None and "quoteprice" in item:
        raw_price = item.get("quoteprice")

    price: Optional[float] = None
    if raw_price is not None:
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = None

    if is_accepted:
        if price is not None and price > 0:
            return True, price, "accepted"
        else:
            return True, None, "eligible_no_price"
    else:
        return False, None, "not_accepted"


async def check_thriftbooks_buyback(isbn: str) -> Tuple[bool, Optional[float], str]:
    """
    Check ThriftBooks buyback status for a single ISBN using their observed endpoint.

    Request (based on your Network tab):
      POST https://www.thriftbooks.com/tb-api/buyback/get-quotes/
      Body:
        {
          "addedFrom": 3,
          "identifiers": ["<ISBN>"]
        }

    This is designed for your personal, low-volume use.
    """
    payload = {
        "addedFrom": 3,
        "identifiers": [isbn],
    }

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.thriftbooks.com",
        "Referer": "https://www.thriftbooks.com/buyback/",
        "User-Agent": "bookshelf-buyback-api/1.0",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                THRIFTBOOKS_QUOTE_URL,
                json=payload,
                headers=headers,
            )
        resp.raise_for_status()
        data = resp.json()

        # Expected: list of quote objects, or dict wrapping such a list
        items: List[dict] = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try a few obvious keys; if they change their shape you'll see "no_data"
            for key, val in data.items():
                if isinstance(val, list):
                    items = val
                    break

        if not items:
            return False, None, "no_data"

        # Prefer an item that matches our ISBN if identifiable
        chosen: Optional[dict] = None
        for item in items:
            if not isinstance(item, dict):
                continue

            identifier = None
            for k, v in item.items():
                lk = k.lower()
                if lk in ("identifier", "isbn", "code"):
                    identifier = str(v).replace("-", "").strip()
                    break

            if identifier is not None and identifier == isbn:
                chosen = item
                break

        if chosen is None:
            # Fallback: if there's only one item, assume it's for our ISBN
            if len(items) == 1 and isinstance(items[0], dict):
                chosen = items[0]

        if not chosen:
            return False, None, "no_match"

        ok, price, status = _extract_tb_offer_from_item(chosen)
        return ok, price, status

    except httpx.HTTPStatusError as e:
        return False, None, f"http_{e.response.status_code}"
    except Exception:
        return False, None, "error"


# ============================================================
# Main endpoint
# ============================================================

@app.post("/check-isbns", response_model=List[ISBNResult])
async def check_isbns(payload: ISBNRequest):
    """
    Accepts a list of ISBNs, normalizes & de-duplicates them,
    checks ThriftBooks buyback, gets metadata from Open Library,
    and returns structured results.
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

        if tb_ok:
            source = "thriftbooks+openlibrary" if (title or author) else "thriftbooks_only"
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
