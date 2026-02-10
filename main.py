import io
import re
import time
import logging
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pytesseract import TesseractNotFoundError

import numpy as np
import cv2
from paddleocr import PaddleOCR

# Point pytesseract to the Tesseract binary inside our Docker image.
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Bookshelf OCR → ISBN Helper")

paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

# ---------------------------
# Performance / reliability knobs
# ---------------------------
MAX_IMAGE_LONG_EDGE = 1100          # Downscale big iPhone photos before OCR
OCR_ANGLES = (0,)                   # whole-image 90° is redundant now that slices rotate
OCR_TIMEOUT_SECONDS = 6             # Hard timeout per OCR call
MIN_OCR_CONF = 25                   # Filter low-confidence junk harder
MAX_CANDIDATES = 10                 # Cap how many candidate lines we try to resolve
OPENLIB_TIMEOUT = 2                 # Keep OpenLibrary fast; failing fast > hanging
TOTAL_SOFT_BUDGET_SECONDS = 25      # If we're over budget, stop resolving more candidates
SPINE_SLICE_COUNT = 6               # how many vertical bands to OCR
SPINE_SLICE_OVERLAP_PX = 40         # overlap so text near edges isn't lost
SPINE_MIN_STRIP_WIDTH = 140         # skip too-thin strips

# Build stamp (lets us confirm the phone is loading the newest HTML)
BUILD_STAMP = "2026-02-10-A"


@app.get("/check-isbns")
def check_isbns():
    return {
        "status": "ok",
        "message": "Bookshelf ISBN helper is running. Use '/' to upload a photo or '/api/bookshelf' for API access.",
    }


# ---------- SMALL HELPERS ----------

def _short(s: str, n: int = 160) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


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


# ---------- OCR HELPERS (Tesseract pipeline you already had) ----------

def extract_ocr_lines(image: Image.Image) -> List[Dict]:
    """
    Spine-optimized OCR:
    1) OCR full image at a couple orientations
    2) Slice into vertical bands, rotate bands upright, OCR each band
    Returns a list of text lines with average confidence.
    """
    all_lines: List[Dict] = []

    def _run_ocr(img: Image.Image, angle_label: str) -> None:
        """Run tesseract, collect line-level text + avg conf."""
        try:
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 3",
                timeout=OCR_TIMEOUT_SECONDS,
            )
        except RuntimeError as e:
            logging.error("OCR timeout/failure (%s): %s", angle_label, e)
            return
        except Exception as e:
            logging.error("OCR failure (%s): %s", angle_label, e)
            return

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
            all_lines.append({"text": full_text, "conf": avg_conf, "angle": angle_label})

    def _prep(img: Image.Image) -> Image.Image:
        """Preprocess for better spine text clarity."""
        g = img.convert("L")
        g = ImageEnhance.Contrast(g).enhance(2.2)
        g = g.filter(ImageFilter.SHARPEN)
        g = g.filter(ImageFilter.MedianFilter(size=3))
        return g

    # --- Pass A: Whole-image OCR (quick)
    for angle in OCR_ANGLES:
        rotated = image.rotate(angle, expand=True)
        _run_ocr(_prep(rotated), angle_label=f"whole_{angle}")

    # --- Pass B: Spine slicing OCR (high value for shelves)
    w, h = image.size
    slice_count = max(6, int(SPINE_SLICE_COUNT))
    base_strip_w = max(1, w // slice_count)

    for i in range(slice_count):
        left = i * base_strip_w - SPINE_SLICE_OVERLAP_PX
        right = (i + 1) * base_strip_w + SPINE_SLICE_OVERLAP_PX
        left = max(0, left)
        right = min(w, right)

        if (right - left) < SPINE_MIN_STRIP_WIDTH:
            continue

        strip = image.crop((left, 0, right, h))

        # Rotate strip so vertical spine text becomes horizontal.
        for rot, label in ((90, "slice_90"),):
            rotated_strip = strip.rotate(rot, expand=True)

            try:
                img = _prep(rotated_strip)
                data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 7 --oem 3",   # single line works well for spines
                    timeout=OCR_TIMEOUT_SECONDS,
                )
            except RuntimeError as e:
                logging.error("OCR timeout/failure (%s): %s", label, e)
                continue
            except Exception as e:
                logging.error("OCR failure (%s): %s", label, e)
                continue

            n = len(data.get("text", []))
            lines: Dict[tuple, Dict] = {}

            for j in range(n):
                raw = data["text"][j]
                if not raw:
                    continue
                text = raw.strip()
                if not text:
                    continue

                conf_str = data.get("conf", ["-1"] * n)[j]
                try:
                    conf = float(conf_str)
                except (ValueError, TypeError):
                    conf = -1.0
                if conf < 0:
                    continue

                key = (
                    data.get("block_num", [0] * n)[j],
                    data.get("par_num", [0] * n)[j],
                    data.get("line_num", [0] * n)[j],
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
                all_lines.append({"text": full_text, "conf": avg_conf, "angle": f"{label}_band{i}"})

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
    return deduped[: max(40, MAX_CANDIDATES * 3)]


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


def normalize_ocr_text(s: str) -> str:
    """
    Repair common OCR spine breakage and normalize for searching.
    """
    s = (s or "").strip().lower()
    s = s.replace("\n", " ")
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = s.replace("collns", "collins")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


COMMON_WORDS = {"the", "and", "of", "in", "on", "to", "for", "with", "a", "an"}


def generate_query_variants(normalized: str) -> List[str]:
    s = (normalized or "").strip().lower()
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    tokens = s.split()

    filtered = []
    for t in tokens:
        if t in COMMON_WORDS:
            filtered.append(t)
            continue
        if len(t) <= 2:
            continue
        if t in {"mm", "rn", "lll", "ii"}:
            continue
        filtered.append(t)

    if not filtered:
        return []

    base = " ".join(filtered)
    drop_last = " ".join(filtered[:-1]).strip() if len(filtered) >= 3 else ""

    merged_tokens = []
    i = 0
    while i < len(filtered):
        if i + 1 < len(filtered):
            if 3 <= len(filtered[i]) <= 5 and 3 <= len(filtered[i + 1]) <= 5:
                merged_tokens.append(filtered[i] + filtered[i + 1])
                i += 2
                continue
        merged_tokens.append(filtered[i])
        i += 1
    merged = " ".join(merged_tokens)

    variants = []
    for v in (base, drop_last, merged):
        v = (v or "").strip()
        if v and v not in variants:
            variants.append(v)

    return variants[:3]


def openlibrary_lookup(query: str, debug_log: Optional[List[Dict]] = None) -> Optional[Dict]:
    q = normalize_ocr_text(query)
    if not q:
        if debug_log is not None:
            debug_log.append({"stage": "normalize", "raw": _short(query), "normalized": ""})
        return None

    variants = generate_query_variants(q)

    if debug_log is not None:
        debug_log.append({"stage": "normalize", "raw": _short(query), "normalized": _short(q)})
        debug_log.append({"stage": "variants", "variants": variants})

    best = None
    best_score = -1

    for qv in variants:
        tokens = qv.split()
        author_guess = ""
        title_guess = qv

        if tokens and tokens[0] in {"collins"} and len(tokens) >= 3:
            author_guess = tokens[0]
            title_guess = " ".join(tokens[1:]).strip()

        attempts = []
        if author_guess and title_guess:
            attempts.append(("structured", {"title": title_guess, "author": author_guess, "limit": 10}))
        if title_guess:
            attempts.append(("title_only", {"title": title_guess, "limit": 10}))
        attempts.append(("q", {"q": qv, "limit": 10}))

        for mode, params in attempts:
            try:
                resp = requests.get(
                    "https://openlibrary.org/search.json",
                    params=params,
                    timeout=OPENLIB_TIMEOUT,
                )
            except Exception as e:
                logging.error("OpenLibrary request failed (%s): %s", mode, e)
                if debug_log is not None:
                    debug_log.append({"stage": "openlibrary", "mode": mode, "params": params, "error": str(e)})
                continue

            if resp.status_code != 200:
                logging.error("OpenLibrary bad status (%s): %s", mode, resp.status_code)
                if debug_log is not None:
                    debug_log.append({"stage": "openlibrary", "mode": mode, "params": params, "status": resp.status_code})
                continue

            data = resp.json()
            docs = data.get("docs") or []

            if debug_log is not None:
                top_docs = []
                for d in docs[:3]:
                    isbns = d.get("isbn") or []
                    top_docs.append({
                        "title": _short(d.get("title") or "", 90),
                        "author": _short(((d.get("author_name") or [""])[0] or ""), 60),
                        "has_isbn": bool(isbns),
                        "isbn_sample": (isbns[0] if isbns else ""),
                    })
                debug_log.append({
                    "stage": "openlibrary",
                    "mode": mode,
                    "params": params,
                    "returned_docs": len(docs),
                    "top_docs": top_docs,
                })

            if not docs:
                continue

            q_words = set(re.findall(r"[a-z0-9]+", qv))

            for d in docs:
                isbns = d.get("isbn") or []
                if not isbns:
                    continue

                title = (d.get("title") or "").strip()
                authors = d.get("author_name") or []
                author = authors[0].strip() if authors else ""

                chosen_isbn = None
                for raw_isbn in isbns:
                    norm = normalize_isbn(raw_isbn)
                    if norm:
                        chosen_isbn = norm
                        break
                if not chosen_isbn:
                    continue

                t_words = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
                a_words = set(re.findall(r"[a-z0-9]+", (author or "").lower()))
                score = len(q_words & t_words) + (1 if ("collins" in q_words and "collins" in a_words) else 0)

                if score > best_score:
                    best_score = score
                    best = {
                        "title": title,
                        "author": author,
                        "isbn13": chosen_isbn,
                        "source": "openlibrary",
                        "match_score": score,
                    }

            if best and best_score >= 2 and mode == "structured":
                break

        if best and best_score >= 2:
            break

    return best


def resolve_candidate(candidate: Dict, debug_log: Optional[List[Dict]] = None) -> Dict:
    query = candidate["raw"]

    ol = openlibrary_lookup(query, debug_log=debug_log)
    if ol:
        ocr_conf = max(0.0, min(candidate["ocr_conf"], 95.0)) / 100.0
        match_norm = 0.3 + 0.1 * (ol.get("match_score", 0))
        blended = max(0.25, min(0.95, 0.45 * ocr_conf + 0.55 * (match_norm / 3.0)))
        return {
            "title": ol["title"],
            "author": ol.get("author", ""),
            "isbn13": ol.get("isbn13", ""),
            "source": "openlibrary",
            "match_score": ol.get("match_score", 0),
            "confidence": round(blended, 2),
        }

    return {
        "title": query,
        "author": "",
        "isbn13": "",
        "source": "unmatched",
        "confidence": round(candidate["ocr_conf"] / 200.0, 2),
    }


# ---------- UI ----------

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
    #debugBox {{ width: 100%; margin-top: 10px; font-size: 0.75rem; padding:8px; box-sizing:border-box; display:none; }}
    label.muted {{ display:block; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Bookshelf → ISBN Helper</h1>
  <p class="muted">Build: {BUILD_STAMP}</p>
  <p>Upload a clear shelf photo. Preview it and rotate if needed before processing.</p>

  <input id="fileCamera" type="file" accept="image/*" capture="environment" style="display:none" />
  <input id="fileLibrary" type="file" accept="image/*" style="display:none" />

  <div class="row">
    <button class="ghost" onclick="chooseCamera()">Take Photo</button>
    <button class="ghost" onclick="chooseLibrary()">Choose from Library</button>
  </div>

  <div id="fileName" class="muted">No photo selected.</div>

  <div class="row">
    <button id="rotLeft" class="ghost" onclick="rotateLeft()" disabled>Rotate ⟲</button>
    <button id="rotRight" class="ghost" onclick="rotateRight()" disabled>Rotate ⟳</button>
    <div class="muted" style="align-self:center;">Rotation: <span id="rotLabel">0°</span></div>
  </div>

  <div style="margin-top:10px; background:#fff; border:1px dashed #d8d2c8; border-radius:10px; padding:10px;">
    <img id="preview" alt="Preview" style="max-width:100%; height:auto; display:none; transform-origin:center center;" />
    <div id="noPreview" class="muted">No preview yet.</div>
  </div>

  <label class="muted">
    <input type="checkbox" id="debugMode" />
    Debug mode (show OCR + lookup details)
  </label>

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

  <textarea id="debugBox" rows="12" readonly></textarea>

<script>
let selectedFile = null;
let currentController = null;
let rotationDegrees = 0;

function setRotation(deg) {{
  rotationDegrees = ((deg % 360) + 360) % 360;
  const img = document.getElementById('preview');
  const label = document.getElementById('rotLabel');
  if (label) label.textContent = rotationDegrees + "°";
  if (img) img.style.transform = `rotate(${{rotationDegrees}}deg)`;
}}

function enableRotateButtons(enabled) {{
  const l = document.getElementById('rotLeft');
  const r = document.getElementById('rotRight');
  if (l) l.disabled = !enabled;
  if (r) r.disabled = !enabled;
}}

function rotateLeft() {{ setRotation(rotationDegrees - 90); }}
function rotateRight() {{ setRotation(rotationDegrees + 90); }}

function setSelectedFile(file) {{
  selectedFile = file || null;

  const fileNameEl = document.getElementById('fileName');
  fileNameEl.textContent = selectedFile ? ("Selected: " + (selectedFile.name || "photo")) : "No photo selected.";

  const img = document.getElementById('preview');
  const noPrev = document.getElementById('noPreview');

  if (!selectedFile) {{
    if (img) {{ img.src = ""; img.style.display = "none"; }}
    if (noPrev) noPrev.style.display = "block";
    enableRotateButtons(false);
    setRotation(0);
    return;
  }}

  setRotation(0);
  enableRotateButtons(true);

  const url = URL.createObjectURL(selectedFile);
  if (img) {{
    img.src = url;
    img.style.display = "block";
  }}
  if (noPrev) noPrev.style.display = "none";
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
  const debugBox = document.getElementById('debugBox');

  const debugOn = document.getElementById('debugMode')?.checked;

  if (!selectedFile) {{
    status.textContent = 'Choose a photo first.';
    return;
  }}

  if (currentController) currentController.abort();
  currentController = new AbortController();

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('debug', debugOn ? '1' : '0');
  formData.append('rotation_degrees', String(rotationDegrees));

  status.textContent = 'Processing... (will time out if it takes too long) — build {BUILD_STAMP}';
  table.style.display = 'none';
  tbody.innerHTML = '';
  isbnBox.style.display = 'none';
  isbnBox.value = '';
  copyBtn.style.display = 'none';
  debugBox.style.display = 'none';
  debugBox.value = '';

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
    const meta = data.meta || null;

    if (debugOn && data.debug) {{
      debugBox.value = JSON.stringify(data.debug, null, 2);
      debugBox.style.display = 'block';
    }}

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
      const elapsed = (meta && meta.elapsed_s != null) ? (" (elapsed " + meta.elapsed_s + "s)") : "";
      status.textContent = "Done. " + isbnList.length + " ISBNs ready to copy. (build {BUILD_STAMP})" + elapsed;
    }} else {{
      const elapsed = (meta && meta.elapsed_s != null) ? (" (elapsed " + meta.elapsed_s + "s)") : "";
      status.textContent = "Done. No ISBNs resolved — try again with a clearer photo." + elapsed;
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
  rotationDegrees = 0;

  document.getElementById('fileName').textContent = 'No photo selected.';
  document.getElementById('status').textContent = '';

  const img = document.getElementById('preview');
  const noPrev = document.getElementById('noPreview');
  if (img) {{ img.src = ""; img.style.display = "none"; img.style.transform = "rotate(0deg)"; }}
  if (noPrev) noPrev.style.display = "block";
  enableRotateButtons(false);
  setRotation(0);

  document.getElementById('resultsTable').style.display = 'none';
  document.getElementById('resultsTable').querySelector('tbody').innerHTML = '';
  document.getElementById('isbnBox').style.display = 'none';
  document.getElementById('isbnBox').value = '';
  document.getElementById('copyBtn').style.display = 'none';

  const debugBox = document.getElementById('debugBox');
  debugBox.style.display = 'none';
  debugBox.value = '';

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
async def process_bookshelf(
    file: UploadFile = File(...),
    debug: str = Form("0"),
    rotation_degrees: int = Form(0),
):
    """
    Core OCR → candidate → lookup pipeline.
    Hardened to avoid long hangs on large photos.
    Debug mode returns extra info (first few lookups + OCR lines).
    """
    started = time.time()
    debug_on = (debug == "1")
    debug_blob = {"ocr_lines": [], "candidates": [], "lookups": []} if debug_on else None

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))

        # Respect iPhone EXIF orientation first, then apply manual rotation from UI.
        image = ImageOps.exif_transpose(image).convert("RGB")
        if rotation_degrees % 360 != 0:
            image = image.rotate(-rotation_degrees, expand=True)

        image = downscale_for_ocr(image)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Unable to read image file."})

    try:
        ocr_lines = extract_ocr_lines(image)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    if debug_on:
        debug_blob["ocr_lines"] = [
            {"text": _short(x.get("text", ""), 120), "conf": round(float(x.get("conf", 0.0)), 1), "angle": x.get("angle", "")}
            for x in (ocr_lines[:25] if ocr_lines else [])
        ]

    candidates = generate_candidates(ocr_lines)

    if debug_on:
        debug_blob["candidates"] = [
            {"raw": _short(c.get("raw", ""), 120), "ocr_conf": round(float(c.get("ocr_conf", 0.0)), 1)}
            for c in candidates
        ]

    books = []
    for idx, cand in enumerate(candidates, start=1):
        if time.time() - started > TOTAL_SOFT_BUDGET_SECONDS:
            logging.warning("Stopping early due to time budget; returning partial results.")
            break

        lookup_log = [] if (debug_on and idx <= 3) else None
        resolved = resolve_candidate(cand, debug_log=lookup_log)

        if debug_on and lookup_log is not None:
            debug_blob["lookups"].append({
                "candidate_raw": _short(cand.get("raw", ""), 140),
                "events": lookup_log,
            })

        books.append(
            {
                "position": idx,
                "title": resolved["title"],
                "author": resolved.get("author", ""),
                "isbn": resolved.get("isbn13", ""),
                "confidence": resolved.get("confidence", 0.0),
                "source": resolved.get("source", "unknown"),
            }
        )

    resp = {
        "books": books,
        "meta": {
            "candidates_used": len(books),
            "elapsed_s": round(time.time() - started, 2),
        },
    }
    if debug_on:
        resp["debug"] = debug_blob

    return resp


# ---------- HEALTH / PADDLE TEST ----------

@app.get("/health/paddle")
def health_paddle():
    try:
        import paddle  # noqa
        from paddleocr import PaddleOCR  # noqa
        import cv2  # noqa
        import numpy as np  # noqa
        return {"ok": True, "paddleocr": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/ocr/paddle")
async def ocr_paddle(
    file: UploadFile = File(...),
    rotation_degrees: int = Form(0),
):
    raw = await file.read()

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")

    if rotation_degrees % 360 != 0:
        img = img.rotate(-rotation_degrees, expand=True)

    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / long_edge
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = paddle_ocr.ocr(bgr, cls=True)

    lines = []
    if result and result[0]:
        for (box, (text, conf)) in result[0]:
            if not text:
                continue
            lines.append({
                "text": " ".join(text.strip().split()),
                "conf": float(conf),
                "box": box,
            })

    def center_xy(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    for ln in lines:
        cx, cy = center_xy(ln["box"])
        ln["cx"] = cx
        ln["cy"] = cy

    lines.sort(key=lambda x: (x["cy"], x["cx"]))

    return {
        "count": len(lines),
        "rotation_used": rotation_degrees,
        "lines": lines,
    }
