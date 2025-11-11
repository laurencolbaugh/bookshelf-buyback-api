import io
import re
import math
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import pytesseract

app = FastAPI(title="Bookshelf OCR → ThriftBooks Helper")

# ---------- OCR HELPERS ----------

def extract_ocr_lines(image: Image.Image) -> List[Dict]:
    """
    Run Tesseract, return a list of lines:
    [
      {"text": "THE HUNGER GAMES", "conf": 92.3},
      ...
    ]
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data["text"])

    lines: Dict[int, Dict] = {}
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1.0

        line_num = data["line_num"][i]
        if line_num not in lines:
            lines[line_num] = {"texts": [], "confs": []}
        lines[line_num]["texts"].append(text)
        lines[line_num]["confs"].append(conf)

    results = []
    for ln, content in lines.items():
        full_text = " ".join(content["texts"]).strip()
        if not full_text:
            continue
        avg_conf = sum(c for c in content["confs"] if c >= 0) / max(
            1, len([c for c in content["confs"] if c >= 0])
        )
        # Filter obvious junk
        if len(full_text) < 3:
            continue
        results.append({"text": full_text, "conf": avg_conf})

    return results


def generate_candidates(ocr_lines: List[Dict]) -> List[Dict]:
    """
    Basic heuristic:
    - Keep lines that look like book-ish text.
    - Later we can get fancier (clusters, orientation, etc.).
    """
    candidates = []
    for line in ocr_lines:
        text = line["text"]

        # Drop lines that look like pure numbers / publisher junk
        if re.fullmatch(r"[0-9\W]+", text):
            continue

        # Drop all-lowercase noise
        if len(text) < 6 and text.islower():
            continue

        candidates.append(
            {
                "raw": text,
                "ocr_conf": line["conf"],
            }
        )

    return candidates


# ---------- LOOKUP HELPERS ----------

def normalize_isbn(isbn: str) -> Optional[str]:
    """Return 13-digit, no-hyphen ISBN if possible."""
    if not isbn:
        return None
    digits = re.sub(r"\D", "", isbn)
    if len(digits) == 13:
        return digits
    if len(digits) == 10:
        # Convert ISBN-10 → ISBN-13
        core = "978" + digits[:-1]
        total = 0
        for i, ch in enumerate(core):
            num = int(ch)
            total += num if i % 2 == 0 else 3 * num
        check = (10 - (total % 10)) % 10
        return core + str(check)
    return None


def thriftbooks_lookup_stub(query: str) -> Optional[Dict]:
    """
    Placeholder for a ThriftBooks-first lookup.

    Intent:
    - Send `query` (title/author-ish) to ThriftBooks search.
    - Take the *closest* match result.
    - Extract canonical title, author, and ISBN13 they list.

    This is left as a stub because:
    - ThriftBooks does not provide a stable, documented public API.
    - Their HTML structure may change.
    - Any scraping must respect their terms of use.

    If you decide to implement:
    - Use requests.get() to hit their search page with `b.search=query`.
    - Parse the first strong match with BeautifulSoup.
    - Grab title, author, and the ISBN/ISBN13 from the product block.
    - Return in the schema below.
    """
    return None  # No-op for now; real logic goes here when you're ready.


def openlibrary_lookup(query: str) -> Optional[Dict]:
    """
    Fallback lookup via OpenLibrary.
    Not ThriftBooks, but gives us real, working metadata today.
    """
    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": 3},
            timeout=6,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    docs = data.get("docs") or []
    if not docs:
        return None

    # Simple: pick the first doc that has an ISBN.
    best = None
    best_score = -1

    for d in docs:
        isbns = d.get("isbn") or []
        if not isbns:
            continue

        title = d.get("title") or ""
        authors = d.get("author_name") or []
        author = authors[0] if authors else ""

        # Prefer 13-digit ISBN; otherwise convert.
        chosen_isbn = None
        for raw in isbns:
            norm = normalize_isbn(raw)
            if norm:
                chosen_isbn = norm
                break

        if not chosen_isbn:
            continue

        # Soft score: longer overlapping words between query and title
        q_words = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
        t_words = set(re.findall(r"[A-Za-z0-9]+", title.lower()))
        overlap = len(q_words & t_words)
        score = overlap

        if score > best_score:
            best_score = score
            best = {
                "title": title,
                "author": author,
                "isbn13": chosen_isbn,
                "source": "openlibrary",
                "match_score": score,
            }

    return best


def resolve_candidate(candidate: Dict) -> Dict:
    """
    ThriftBooks-first resolution, with OpenLibrary fallback.
    Returns a unified record with confidence.
    """
    query = candidate["raw"]

    # 1) Try ThriftBooks (stubbed for now)
    tb = thriftbooks_lookup_stub(query)
    if tb:
        # You can tune these once real TB data is wired in
        return {
            "title": tb["title"],
            "author": tb.get("author", ""),
            "isbn": tb.get("isbn13") or "",
            "source": "thriftbooks",
            "confidence": min(0.99, 0.6 + candidate["ocr_conf"] / 100 * 0.4),
        }

    # 2) Fallback: OpenLibrary
    ol = openlibrary_lookup(query)
    if ol:
        # Blend OCR confidence and match_score (very rough)
        ocr_conf = max(0.0, min(candidate["ocr_conf"], 95.0)) / 100.0
        match_norm = 0.3 + 0.1 * (ol.get("match_score", 0))
        blended = max(0.3, min(0.98, 0.4 * ocr_conf + 0.6 * (match_norm / 3.0)))

        return {
            "title": ol["title"],
            "author": ol.get("author", ""),
            "isbn": ol.get("isbn13", ""),
            "source": ol.get("source", "openlibrary"),
            "confidence": round(blended, 2),
        }

    # 3) No match: return raw text, low confidence, blank ISBN
    return {
        "title": query,
        "author": "",
        "isbn": "",
        "source": "unmatched",
        "confidence": round(candidate["ocr_conf"] / 200.0, 2),  # 0–0.5 range-ish
    }


# ---------- API ENDPOINTS ----------

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Minimal phone-friendly UI:
    - Upload photo
    - See table
    - Copy ISBN column with one click
    """
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bookshelf → ThriftBooks ISBN Helper</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; padding: 16px; background:#faf7f2; color:#222; }
    h1 { font-size: 1.3rem; margin-bottom: 0.3rem; }
    p { font-size: 0.9rem; margin-top: 0; margin-bottom: 0.8rem; }
    input[type=file] { margin: 8px 0 12px; }
    button { padding: 8px 14px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
    button.primary { background: #4b6b5c; color: #fff; }
    button.secondary { background: #e6e2db; color: #333; margin-left: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-top: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
    th { background: #f1ece4; }
    #status { font-size: 0.8rem; margin-top: 6px; color:#555; }
    #isbnBox { width: 100%; margin-top: 8px; font-size: 0.75rem; padding:6px; }
  </style>
</head>
<body>
  <h1>Bookshelf → ThriftBooks ISBN Helper</h1>
  <p>Upload a clear shelf photo. I’ll OCR the spines, resolve against book data, and give you a clean ISBN column to paste into ThriftBooks buyback.</p>

  <input id="file" type="file" accept="image/*" capture="environment" />
  <br/>
  <button class="primary" onclick="processImage()">Process Photo</button>
  <button class="secondary" onclick="clearAll()">Clear</button>

  <div id="status"></div>

  <table id="resultsTable" style="display:none;">
    <thead>
      <tr>
        <th>#</th>
        <th>Title</th>
        <th>Author</th>
        <th>ISBN (13)</th>
        <th>Confidence</th>
        <th>Source</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <textarea id="isbnBox" rows="4" readonly style="display:none;"></textarea>
  <button id="copyBtn" class="secondary" style="display:none;" onclick="copyIsbns()">Copy ISBN Column</button>

<script>
async function processImage() {
  const fileInput = document.getElementById('file');
  const status = document.getElementById('status');
  const table = document.getElementById('resultsTable');
  const tbody = table.querySelector('tbody');
  const isbnBox = document.getElementById('isbnBox');
  const copyBtn = document.getElementById('copyBtn');

  if (!fileInput.files.length) {
    status.textContent = 'Choose a photo first.';
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  status.textContent = 'Processing...';
  table.style.display = 'none';
  tbody.innerHTML = '';
  isbnBox.style.display = 'none';
  isbnBox.value = '';
  copyBtn.style.display = 'none';

  try {
    const resp = await fetch('/api/bookshelf', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) {
      status.textContent = 'Error from server. Try a clearer photo.';
      return;
    }

    const data = await resp.json();
    const books = data.books || [];

    if (!books.length) {
      status.textContent = 'No candidates detected. Try a closer or clearer shot.';
      return;
    }

    let isbnList = [];
    books.forEach((b, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${(b.title || '').replace(/</g,'&lt;')}</td>
        <td>${(b.author || '').replace(/</g,'&lt;')}</td>
        <td>${b.isbn || ''}</td>
        <td>${(b.confidence != null) ? (b.confidence * 100).toFixed(0) + '%' : ''}</td>
        <td>${b.source || ''}</td>
      `;
      tbody.appendChild(tr);
      if (b.isbn) {
        isbnList.push(b.isbn);
      }
    });

    table.style.display = 'table';

    if (isbnList.length) {
      isbnBox.value = isbnList.join('\\n');
      isbnBox.style.display = 'block';
      copyBtn.style.display = 'inline-block';
      status.textContent = `Done. ${isbnList.length} ISBNs ready to copy.`;
    } else {
      status.textContent = 'Done. No ISBNs resolved — review titles and try again.';
    }
  } catch (e) {
    console.error(e);
    status.textContent = 'Unexpected error. Try again.';
  }
}

function clearAll() {
  document.getElementById('file').value = '';
  document.getElementById('status').textContent = '';
  document.getElementById('resultsTable').style.display = 'none';
  document.getElementById('resultsTable').querySelector('tbody').innerHTML = '';
  document.getElementById('isbnBox').style.display = 'none';
  document.getElementById('isbnBox').value = '';
  document.getElementById('copyBtn').style.display = 'none';
}

async function copyIsbns() {
  const isbnBox = document.getElementById('isbnBox');
  isbnBox.select();
  isbnBox.setSelectionRange(0, 99999);
  try {
    await navigator.clipboard.writeText(isbnBox.value);
    document.getElementById('status').textContent = 'ISBNs copied to clipboard.';
  } catch {
    document.getElementById('status').textContent = 'Select + copy manually (clipboard blocked).';
  }
}
</script>
</body>
</html>
    """


@app.post("/api/bookshelf")
async def process_bookshelf(file: UploadFile = File(...)):
    """
    Core pipeline:
    1. Load image
    2. OCR → lines
    3. Heuristic candidates
    4. ThriftBooks-first (stub) / OpenLibrary fallback
    5. Return structured list suitable for your table + ISBN paste
    """
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Unable to read image file."},
        )

    ocr_lines = extract_ocr_lines(image)
    candidates = generate_candidates(ocr_lines)

    books = []
    pos = 1
    for cand in candidates:
        resolved = resolve_candidate(cand)
        books.append(
            {
                "position": pos,
                "title": resolved["title"],
                "author": resolved.get("author", ""),
                "isbn": resolved.get("isbn", ""),
                "confidence": resolved.get("confidence", 0.0),
                "source": resolved.get("source", "unknown"),
            }
        )
        pos += 1

    return {"books": books}


# To run locally:
# uvicorn main:app --host 0.0.0.0 --port 8000
