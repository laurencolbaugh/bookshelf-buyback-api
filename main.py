import io
import re
import time
import logging
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pytesseract import TesseractNotFoundError

# Point pytesseract to the Tesseract binary inside our Docker image.
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Bookshelf OCR → ISBN Helper")

# ---------------------------
# Performance / reliability knobs
# ---------------------------
MAX_IMAGE_LONG_EDGE = 1600          # Downscale big iPhone photos before OCR
OCR_ANGLES = (0, 90)                # Fewer rotations = faster; 0 + 90 catches most spines
OCR_TIMEOUT_SECONDS = 12            # Hard timeout per OCR call
MIN_OCR_CONF = 25                   # Filter low-confidence junk harder
MAX_CANDIDATES = 18                 # Cap how many candidate lines we try to resolve
OPENLIB_TIMEOUT = 4                 # Keep OpenLibrary fast; failing fast > hanging
TOTAL_SOFT_BUDGET_SECONDS = 25      # If we're over budget, stop resolving more candidates

# Build stamp (lets us confirm the phone is loading the newest HTML)
BUILD_STAMP = "2026-02-07-B"


@app.get("/check-isbns")
def check_isbns():
    return {
        "status": "ok",
        "message": "Bookshelf ISBN helper is running. Use '/' to upload a photo or '/api/bookshelf' for API access.",
    }


# ---------- IMAGE PREP ----------

def downscale_for_ocr(image: Image.Image) -> Image.Image:
    """
    Downscale large images to keep OCR fast and avoid Render timeouts.
    """
    image = ImageOps.exif_transpose(image)  # respect iPhone orientation metadata
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= MAX_IMAGE_LONG_EDGE:
        return image

    scale = MAX_IMAGE_LONG_EDGE / float(long_edge)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logging.info("Downscaling image from %sx%s to %sx%s for OCR", w, h, new_w, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------- OCR HELPERS ----------

def extract_ocr_lines(image: Image.Image) -> List[Dict]:
    """
    Preprocess + multi-orientation OCR.
    Returns a list of text lines with average confidence.
    """
    all_lines: List[Dict] = []

    for angle in OCR_ANGLES:
        try:
            rotated = image.rotate(angle, expand=True)

            # Preprocess for better text clarity
            img = rotated.convert("L")  # grayscale
            img = ImageEnhance.Contrast(img).enhance(1.5)


            # timeout prevents "infinite" OCR on hard images
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 3",
                timeout=OCR_TIMEOUT_SECONDS,
            )
        except TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR not found in environment.")
        except RuntimeError as e:
            # pytesseract raises RuntimeError on timeout
            logging.error("OCR timeout/failure at angle %s: %s", angle, e)
            continue
        except Exception as e:
            logging.error("OCR failure at angle %s: %s", angle, e)
            continue

        n = len(data.get("text", []))
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
            avg_conf = sum(content["confs"]) / max(1, len(content["confs"]))
            all_lines.append({"text": full_text, "conf": avg_conf, "angle": angle})

    if not all_lines:
        return []

    # Deduplicate & keep only reasonable-confidence lines
    deduped: List[Dict] = []
    seen = set()
    for entry in sorted(all_lines, key=lambda x: x["conf"], reverse=True):
        if entry["conf"] < MIN_OCR_CONF:
            continue
        text = entry["text"].strip()
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"text": text, "conf": entry["conf"]})

    # Keep only top N lines to avoid explosion
    return deduped[: max(30, MAX_CANDIDATES * 2)]


# ---------- CANDIDATE GENERATION ----------

NOISE_TERMS = [
    "isbn", "press", "edition", "publish", "printing",
    "inc", "llc", "www", ".com"
]


def clean_line_to_titleish(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", text)
    if len(tokens) < 2:
        return None

    filtered = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            continue
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
    candidates: List[Dict] = []
    for line in ocr_lines:
        cleaned = clean_line_to_titleish(line["text"])
        if not cleaned:
            continue
        candidates.append({"raw": cleaned, "ocr_conf": line["conf"]})

    # Sort by OCR confidence, then cap count
    candidates.sort(key=lambda c: c["ocr_conf"], reverse=True)
    return candidates[:MAX_CANDIDATES]


# ---------- LOOKUP HELPERS ----------

def normalize_isbn(isbn: str) -> Optional[str]:
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


def openlibrary_lookup(query: str) -> Optional[Dict]:
    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": 5},
            timeout=OPENLIB_TIMEOUT,
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

    q_words = set(re.findall(r"[A-Za-z0-9]+", query.lower()))

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

        t_words = set(re.findall(r"[A-Za-z0-9]+", title.lower()))
        score = len(q_words & t_words)

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
    query = candidate["raw"]

    ol = openlibrary_lookup(query)
    if ol:
        # Conservative confidence blending
        ocr_conf = max(0.0, min(candidate["ocr_conf"], 95.0)) / 100.0
        match_norm = 0.3 + 0.1 * (ol.get("match_score", 0))
        blended = max(0.25, min(0.95, 0.45 * ocr_conf + 0.55 * (match_norm / 3.0)))
        return {
            "title": ol["title"],
            "author": ol.get("author", ""),
            "isbn": ol.get("isbn13", ""),
            "source": "openlibrary",
            "confidence": round(blended, 2),
        }

    return {
        "title": query,
        "author": "",
        "isbn": "",
        "source": "unmatched",
        "confidence": round(candidate["ocr_conf"] / 200.0, 2),
    }


# ---------- UI ----------
# IMPORTANT CHANGE:
# Return HTML with NO-CACHE headers so iPhone Safari can’t keep serving an old copy of the page.

@app.get("/")
def index():
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bookshelf → ISBN Helper</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; padding: 16px; background:#faf7f2; color:#222; }}
    h1 {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
    p {{ font-size: 0.9rem; margin-top: 0; margin-bottom: 0.8rem; }}
    button {{ padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; }}
    button.primary {{ background: #4b6b5c; color: #fff; }}
    button.secondary {{ background: #e6e2db; color: #333; margin-left: 8px; }}
    button.ghost {{ background: #fff; color:#333; border:1px solid #d8d2c8; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 12px; }}
    .row button {{ flex: 1 1 160px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
    th {{ background: #f1ece4; }}
    #status {{ font-size: 0.85rem; margin-top: 8px; color:#555; white-space: pre-line; }}
    #isbnBox {{ width: 100%; margin-top: 8px; font-size: 0.8rem; padding:8px; box-sizing:border-box; }}
    #fileName {{ font-size: 0.85rem; color:#333; margin-top: 6px; }}
    .muted {{ color:#666; font-size:0.85rem; }}
  </style>
</head>
<body>
  <h1>Bookshelf → ISBN Helper</h1>
  <p class="muted">Build: {BUILD_STAMP}</p>
  <p>Upload a clear shelf photo. If it takes too long, the app will time out instead of hanging forever.</p>

  <input id="fileCamera" type="file" accept="image/*" capture="environment" style="display:none" />
  <input id="fileLibrary" type="file" accept="image/*" style="display:none" />

  <div class="row">
    <button class="ghost" onclick="chooseCamera()">Take Photo</button>
    <button class="ghost" onclick="chooseLibrary()">Choose from Library</button>
  </div>

  <div id="fileName" class="muted">No photo selected.</div>

  <div class="row">
    <button class="primary" onclick="processImage()">Process Photo</button>
    <button class="secondary" onclick="clearAll()">Clear</button>
  </div>

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
let selectedFile = null;
let currentController = null;

function setSelectedFile(file) {{
  selectedFile = file || null;
  const fileNameEl = document.getElementById('fileName');
  fileNameEl.textContent = selectedFile ? ("Selected: " + (selectedFile.name || "photo")) : "No photo selected.";
}}

function chooseCamera() {{ document.getElementById('fileCamera').click(); }}
function chooseLibrary() {{ document.getElementById('fileLibrary').click(); }}

document.getElementById('fileCamera').addEventListener('change', (e) => {{
  setSelectedFile(e.target.files && e.target.files[0]);
}});
document.getElementById('fileLibrary').addEventListener('change', (e) => {{
  setSelectedFile(e.target.files && e.target.files[0]);
}});

async function processImage() {{
  const status = document.getElementById('status');
  const table = document.getElementById('resultsTable');
  const tbody = table.querySelector('tbody');
  const isbnBox = document.getElementById('isbnBox');
  const copyBtn = document.getElementById('copyBtn');

  if (!selectedFile) {{
    status.textContent = 'Choose a photo first.';
    return;
  }}

  // Cancel any prior request
  if (currentController) currentController.abort();
  currentController = new AbortController();

  const formData = new FormData();
  formData.append('file', selectedFile);

  status.textContent = 'Processing... (will time out if it takes too long) — build {BUILD_STAMP}';
  table.style.display = 'none';
  tbody.innerHTML = '';
  isbnBox.style.display = 'none';
  isbnBox.value = '';
  copyBtn.style.display = 'none';

  // Hard client timeout so UI never hangs forever
  const timeoutMs = 35000;
  const timeoutId = setTimeout(() => currentController.abort(), timeoutMs);

  try {{
    const resp = await fetch('/api/bookshelf', {{
      method: 'POST',
      body: formData,
      signal: currentController.signal
    }});

    if (!resp.ok) {{
      const errText = await resp.text().catch(() => '');
      console.error('Server error:', errText);
      status.textContent = 'Server returned an error. Try a closer / clearer photo.';
      return;
    }}

    const data = await resp.json();
    const books = data.books || [];

    if (!books.length) {{
      status.textContent = 'No candidates detected. Try a closer or clearer shot.';
      return;
    }}

    let isbnList = [];
    books.forEach((b, idx) => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{idx + 1}}</td>
        <td>${{(b.title || '').replace(/</g,'&lt;')}}</td>
        <td>${{(b.author || '').replace(/</g,'&lt;')}}</td>
        <td>${{b.isbn || ''}}</td>
        <td>${{(b.confidence != null) ? (b.confidence * 100).toFixed(0) + '%' : ''}}</td>
        <td>${{b.source || ''}}</td>
      `;
      tbody.appendChild(tr);
      if (b.isbn) isbnList.push(b.isbn);
    }});

    table.style.display = 'table';

    if (isbnList.length) {{
      isbnBox.value = isbnList.join('\\n');
      isbnBox.style.display = 'block';
      copyBtn.style.display = 'inline-block';
      status.textContent = `Done. ${{isbnList.length}} ISBNs ready to copy. (build {BUILD_STAMP})`;
    }} else {{
      status.textContent = 'Done. No ISBNs resolved — try again with a clearer photo.';
    }}
  }} catch (e) {{
    if (e.name === 'AbortError') {{
      status.textContent = 'Timed out. Try a closer photo, or crop to fewer books per image.';
    }} else {{
      console.error(e);
      status.textContent = 'Unexpected error. Try again.';
    }}
  }} finally {{
    clearTimeout(timeoutId);
  }}
}}

function clearAll() {{
  document.getElementById('fileCamera').value = '';
  document.getElementById('fileLibrary').value = '';
  selectedFile = null;
  document.getElementById('fileName').textContent = 'No photo selected.';
  document.getElementById('status').textContent = '';
  document.getElementById('resultsTable').style.display = 'none';
  document.getElementById('resultsTable').querySelector('tbody').innerHTML = '';
  document.getElementById('isbnBox').style.display = 'none';
  document.getElementById('isbnBox').value = '';
  document.getElementById('copyBtn').style.display = 'none';
  if (currentController) currentController.abort();
}}

async function copyIsbns() {{
  const isbnBox = document.getElementById('isbnBox');
  isbnBox.select();
  isbnBox.setSelectionRange(0, 99999);
  try {{
    await navigator.clipboard.writeText(isbnBox.value);
    document.getElementById('status').textContent = 'ISBNs copied to clipboard.';
  }} catch {{
    document.getElementById('status').textContent = 'Select + copy manually (clipboard blocked).';
  }}
}}
</script>
</body>
</html>
    """

    # No-cache headers: prevents iPhone Safari from reusing an old copy of "/"
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------- API ENDPOINT ----------

@app.post("/api/bookshelf")
async def process_bookshelf(file: UploadFile = File(...)):
    """
    Core OCR → candidate → lookup pipeline.
    Hardened to avoid long hangs on large photos.
    """
    started = time.time()

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB")
        image = downscale_for_ocr(image)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Unable to read image file."})

    try:
        ocr_lines = extract_ocr_lines(image)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    candidates = generate_candidates(ocr_lines)

    books = []
    for idx, cand in enumerate(candidates, start=1):
        # Soft budget: stop doing lookups if we're taking too long
        if time.time() - started > TOTAL_SOFT_BUDGET_SECONDS:
            logging.warning("Stopping early due to time budget; returning partial results.")
            break

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

    return {"books": books, "meta": {"candidates_used": len(books), "elapsed_s": round(time.time() - started, 2)}}

# To run locally:
# uvicorn main:app --host 0.0.0.0 --port 8000

