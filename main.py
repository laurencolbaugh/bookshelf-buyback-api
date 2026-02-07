import io
import re
import logging
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import TesseractNotFoundError

# Point pytesseract to the Tesseract binary inside our Docker image.
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Bookshelf OCR → ThriftBooks Helper")


@app.get("/check-isbns")
def check_isbns():
    return {
        "status": "ok",
        "message": "Bookshelf ISBN helper is running. Use '/' to upload a photo or '/api/bookshelf' for API access.",
    }


# ---------- OCR HELPERS ----------

def extract_ocr_lines(image: Image.Image) -> List[Dict]:
    """
    Preprocess + multi-orientation OCR.
    Returns a list of text lines with average confidence.
    """
    all_lines: List[Dict] = []

    for angle in (0, 90, 270):
        try:
            rotated = image.rotate(angle, expand=True)

            # Preprocess for better text clarity
            img = rotated.convert("L")  # grayscale
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            img = img.filter(ImageFilter.MedianFilter(size=3))

            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 3",
            )
        except TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR not found in environment.")
        except Exception as e:
            logging.error("OCR failure at angle %s: %s", angle, e)
            continue

        n = len(data["text"])
        lines: Dict[tuple, Dict] = {}

        for i in range(n):
            raw = data["text"][i]
            if not raw:
                continue

            text = raw.strip()
            if not text:
                continue

            conf_str = data.get("conf", ["-1"] * n)[i]
            try:
                conf = float(conf_str)
            except (ValueError, TypeError):
                conf = -1.0
            if conf < 0:
                continue

            key = (
                data.get("block_num", [0] * n)[i],
                data.get("par_num", [0] * n)[i],
                data.get("line_num", [0] * n)[i],
            )
            if key not in lines:
                lines[key] = {"texts": [], "confs": []}
            lines[key]["texts"].append(text)
            lines[key]["confs"].append(conf)

        for _, content in lines.items():
            full_text = " ".join(content["texts"]).strip()
            if len(full_text) < 3:
                continue
            avg_conf = sum(content["confs"]) / len(content["confs"])
            all_lines.append(
                {
                    "text": full_text,
                    "conf": avg_conf,
                    "angle": angle,
                }
            )

    if not all_lines:
        try:
            raw = pytesseract.image_to_string(image)
            logging.info("[DEBUG] Fallback OCR sample: %s", raw[:200])
        except Exception:
            pass
        return []

    # Deduplicate & keep only reasonable-confidence lines
    deduped: List[Dict] = []
    seen = set()
    for entry in sorted(all_lines, key=lambda x: x["conf"], reverse=True):
        text = entry["text"].strip()
        if entry["conf"] < 15:  # toss ultra-low confidence
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"text": text, "conf": entry["conf"]})

    if len(deduped) <= 5:
        logging.info("[DEBUG] OCR lines after filtering: %s", [d["text"] for d in deduped])

    return deduped


# ---------- CANDIDATE GENERATION ----------

NOISE_TERMS = [
    "isbn", "press", "edition", "publish", "printing",
    "inc", "llc", "www", ".com"
]


def clean_line_to_titleish(text: str) -> Optional[str]:
    """
    Take a noisy OCR line and keep only plausible word tokens.
    Returns cleaned string or None if it's junk.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", text)

    if len(tokens) < 2:
        return None  # need at least two tokens to be interesting

    filtered = []
    for t in tokens:
        if len(t) <= 2 and t.isupper():
            continue
        filtered.append(t)

    if len(filtered) < 2:
        return None

    cleaned = " ".join(filtered)
    lower = cleaned.lower()

    if any(term in lower for term in NOISE_TERMS):
        return None

    if not any(len(w) >= 4 for w in filtered):
        return None

    return cleaned.strip()


def generate_candidates(ocr_lines: List[Dict]) -> List[Dict]:
    """
    Turn OCR lines into cleaned candidate strings for lookup.
    """
    candidates: List[Dict] = []

    for line in ocr_lines:
        cleaned = clean_line_to_titleish(line["text"])
        if not cleaned:
            continue

        candidates.append(
            {
                "raw": cleaned,
                "ocr_conf": line["conf"],
            }
        )

    logging.info("[DEBUG] Candidates: %s", [c["raw"] for c in candidates])

    return candidates


# ---------- LOOKUP HELPERS ----------

def normalize_isbn(isbn: str) -> Optional[str]:
    """
    Return 13-digit, no-hyphen ISBN if possible.
    """
    if not isbn:
        return None
    digits = re.sub(r"\D", "", isbn)
    if len(digits) == 13:
        return digits
    if len(digits) == 10:
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
    Placeholder: ThriftBooks-first lookup would go here.
    """
    return None


def openlibrary_lookup(query: str) -> Optional[Dict]:
    """
    Fallback lookup via OpenLibrary.
    """
    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": 5},
            timeout=6,
        )
    except Exception as e:
        logging.error("OpenLibrary request failed: %s", e)
        return None

    if resp.status_code != 200:
        logging.error("OpenLibrary bad status: %s", resp.status_code)
        return None

    data = resp.json()
    docs = data.get("docs") or []
    if not docs:
        return None

    best = None
    best_score = -1

    for d in docs:
        isbns = d.get("isbn") or []
        if not isbns:
            continue

        title = (d.get("title") or "").strip()
        authors = d.get("author_name") or []
        author = authors[0].strip() if authors else ""

        chosen_isbn = None
        for raw in isbns:
            norm = normalize_isbn(raw)
            if norm:
                chosen_isbn = norm
                break
        if not chosen_isbn:
            continue

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
    Try ThriftBooks (stub), then OpenLibrary; otherwise return unmatched.
    """
    query = candidate["raw"]

    tb = thriftbooks_lookup_stub(query)
    if tb:
        return {
            "title": tb["title"],
            "author": tb.get("author", ""),
            "isbn": tb.get("isbn13") or "",
            "source": "thriftbooks",
            "confidence": min(0.99, 0.6 + candidate["ocr_conf"] / 100 * 0.4),
        }

    ol = openlibrary_lookup(query)
    if ol:
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

    return {
        "title": query,
        "author": "",
        "isbn": "",
        "source": "unmatched",
        "confidence": round(candidate["ocr_conf"] / 200.0, 2),
    }


# ---------- API ENDPOINTS ----------

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Minimal phone-friendly UI: upload → table → copy ISBNs.
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

  <!-- FIX: removed capture="environment" so iPhone offers Photo Library -->
  <input id="file" type="file" accept="image/*" />
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
      const errText = await resp.text().catch(() => '');
      console.error('Server error:', errText);
      status.textContent = 'Error from server. Try a clearer photo or check logs.';
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
    Core OCR → candidate → lookup pipeline.
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

    try:
        ocr_lines = extract_ocr_lines(image)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    candidates = generate_candidates(ocr_lines)

    books = []
    for idx, cand in enumerate(candidates, start=1):
        resolved = resolve_candidate(cand)
        books.append(
            {
                "position": idx,
                "title": resolved["title"],
                "author": resolved.get("author", ""),
                "isbn": resolved.get("isbn", ""),
                "confidence": resolved.get("confidence", 0.0),
                "source": resolved.get("source", "unknown"),
            }
        )

    return {"books": books}

# To run locally:
# uvicorn main:app --host 0.0.0.0 --port 8000
