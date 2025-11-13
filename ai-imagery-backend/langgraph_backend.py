# langgraph_backend.py
"""
LangGraph + Gemini Backend — Ultra-Stable Fallback edition

Behavior:
- Try real Gemini calls once (identify, select_best, segment, enhance) when USE_REAL_GEMINI=True and GOOGLE_API_KEY is present.
- If any real call fails or returns unusable content -> immediate deterministic fallback (no retry).
- Frame extraction limited to 3 frames at 20%, 50%, 80%.
- MAX_PRODUCTS and VARIATIONS configurable via env vars.
- RUN_ID isolation and output cleanup to avoid mixups.
- Simple file-serving endpoint with no-store header.
- CORS enabled for local frontend.
"""

import os
import io
import json
import shutil
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------------------------
# Config (env)
# ---------------------------
USE_REAL_GEMINI = os.environ.get("USE_REAL_GEMINI", "true").lower() in ("1", "true", "yes")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)

MAX_PRODUCTS = int(os.environ.get("MAX_PRODUCTS", "2"))
VARIATIONS = int(os.environ.get("VARIATIONS", "3"))
LG_WORKDIR = Path(os.environ.get("LG_WORKDIR", "./workdir")).absolute()

GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "models/gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "models/gemini-2.5-flash")

# Ensure directories
VIDEO_DIR = LG_WORKDIR / "videos"
FRAMES_DIR = LG_WORKDIR / "frames"
OUTPUT_DIR = LG_WORKDIR / "output"
for d in (VIDEO_DIR, FRAMES_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Gemini SDK init (if available)
# ---------------------------
genai = None
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
        except Exception as e:
            print("[WARN] genai.configure() failed:", e)
    except Exception:
        genai = None
        print("[WARN] google.generativeai SDK not installed or import failed. Falling back to mock behavior.")
else:
    if USE_REAL_GEMINI:
        print("[WARN] USE_REAL_GEMINI=True but GOOGLE_API_KEY not set. Falling back to mock mode.")
    USE_REAL_GEMINI = False

# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI(title="LangGraph + Gemini Backend (fallback-first)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    youtube_url: str
    max_products: Optional[int] = MAX_PRODUCTS
    frames_sample_rate: Optional[float] = 1.0

# ---------------------------
# Utilities
# ---------------------------
def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def reset_output_dir():
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except Exception:
                pass
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_youtube_video(youtube_url: str, run_id: str) -> Path:
    safe_makedirs(VIDEO_DIR)
    out_path = VIDEO_DIR / f"video_{run_id}.mp4"
    if out_path.exists():
        return out_path

    if shutil.which("yt-dlp"):
        import subprocess
        cmd = ["yt-dlp", "-f", "best[ext=mp4]/mp4", "-o", str(out_path), youtube_url]
        subprocess.check_call(cmd)
        return out_path

    try:
        from pytube import YouTube
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream.download(output_path=str(VIDEO_DIR), filename=out_path.name)
        return out_path
    except Exception as e:
        raise RuntimeError(f"video download failed: {e}")

def extract_three_frames_at_percentiles(video_path: Path) -> List[Path]:
    frames_folder = FRAMES_DIR / video_path.stem
    safe_makedirs(frames_folder)
    found = sorted(frames_folder.glob("frame_*.jpg"))
    if found and len(found) >= 3:
        return found[:3]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for frame extraction")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0 or frame_count == 0:
        saved = []
        while len(saved) < 3:
            ret, frame = cap.read()
            if not ret:
                break
            out = frames_folder / f"frame_{len(saved):06d}.jpg"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out, quality=90)
            saved.append(out)
        cap.release()
        return saved

    percentiles = [0.2, 0.5, 0.8]
    saved = []
    for i, p in enumerate(percentiles):
        t = max(0.0, min(duration * p, duration - 0.01))
        frame_no = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_no, frame_count - 1))
            ret, frame = cap.read()
            if not ret:
                continue
        out = frames_folder / f"frame_{i:06d}.jpg"
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out, quality=90)
        saved.append(out)
    cap.release()
    return saved

# ---------------------------
# Gemini helpers (single-attempt)
# ---------------------------

def safe_generate_once(model_obj, inputs) -> Optional[Any]:
    """
    Attempt a single model call. If it raises or returns unusable output, return None.
    No retries, no sleeps.
    """
    if model_obj is None:
        return None
    try:
        # Try common method names
        if hasattr(model_obj, "generate_content"):
            return model_obj.generate_content(inputs)
        if hasattr(model_obj, "predict"):
            return model_obj.predict(inputs)
        # Some SDKs may wrap model differently
        return None
    except Exception as e:
        print(f"[WARN] model call failed (single attempt): {e}")
        return None

def extract_text_from_response(resp) -> str:
    if resp is None:
        return ""
    try:
        return getattr(resp, "text", "") or ""
    except Exception:
        pass
    try:
        if hasattr(resp, "candidates"):
            for cand in resp.candidates:
                if hasattr(cand, "content") and getattr(cand.content, "parts", None):
                    return cand.content.parts[0].text or ""
    except Exception:
        pass
    try:
        if getattr(resp, "parts", None):
            for p in resp.parts:
                if getattr(p, "text", None):
                    return p.text
    except Exception:
        pass
    try:
        return str(resp)
    except Exception:
        return ""

# ---------------------------
# Pipeline step functions (single-attempt + immediate fallback)
# ---------------------------

async def identify_products_in_frame(frame_path: Path) -> List[Dict[str, Any]]:
    print(f"[NODE] identify -> {frame_path}")
    if not USE_REAL_GEMINI or genai is None:
        return [{"label": "product", "bbox": None}]

    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    except Exception:
        print("[WARN] couldn't initialize genai model object for identify; using fallback")
        return [{"label": "product", "bbox": None}]

    img = Image.open(frame_path).convert("RGB")
    prompt = "Identify visible product objects in the image. Return a JSON list of {label, confidence, bbox|null}."
    resp = safe_generate_once(model, [prompt, img])
    text = extract_text_from_response(resp).strip()
    if not text:
        print("[WARN] identify returned no usable text; fallback to generic product")
        return [{"label": "product", "bbox": None}]
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start:end+1])
            if isinstance(parsed, list) and parsed:
                out = []
                for p in parsed:
                    if isinstance(p, dict) and "label" in p:
                        out.append({"label": p.get("label"), "bbox": p.get("bbox", None)})
                return out[:MAX_PRODUCTS]
    except Exception as e:
        print(f"[WARN] parse identify JSON failed: {e}")
    return [{"label": "product", "bbox": None}]

async def select_best_frame_for_product(candidate_frames: List[Path], product_label: str) -> Path:
    if not candidate_frames:
        raise ValueError("no candidate frames")
    if not USE_REAL_GEMINI or genai is None or len(candidate_frames) == 1:
        return candidate_frames[0]

    print(f"[NODE] select_best -> label={product_label} candidates={len(candidate_frames)}")
    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    except Exception:
        print("[WARN] select_best: couldn't init model; fallback to first")
        return candidate_frames[0]

    prompt = f"From the images, pick the filename where product '{product_label}' is shown most prominently. Reply with filename only."
    imgs = [Image.open(p).convert("RGB") for p in candidate_frames]
    resp = safe_generate_once(model, [prompt] + imgs)
    text = extract_text_from_response(resp)
    if not text:
        print("[WARN] select_best returned no usable text; fallback to first candidate")
        return candidate_frames[0]
    for p in candidate_frames:
        if p.name in text:
            return p
    return candidate_frames[len(candidate_frames)//2]

async def segment_product_in_frame(frame_path: Path, run_id: str, bbox: Optional[Dict[str,int]] = None) -> Path:
    out_png = OUTPUT_DIR / f"seg_{run_id}_{frame_path.stem}.png"
    print(f"[NODE] segment -> best={frame_path}")

    if not USE_REAL_GEMINI or genai is None:
        # deterministic center-crop fallback
        img_cv = cv2.imread(str(frame_path))
        if img_cv is None:
            raise RuntimeError("failed to read frame for segmentation fallback")
        h, w = img_cv.shape[:2]
        cx, cy = w//2, h//2
        bw, bh = int(w * 0.6), int(h * 0.6)
        x0 = max(0, cx - bw//2)
        y0 = max(0, cy - bh//2)
        crop = img_cv[y0:y0+bh, x0:x0+bw]
        cv2.imwrite(str(out_png), crop)
        return out_png

    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    except Exception:
        print("[WARN] segmentation: couldn't init model; using fallback crop")
        img_cv = cv2.imread(str(frame_path))
        h, w = img_cv.shape[:2]
        cx, cy = w//2, h//2
        bw, bh = int(w * 0.6), int(h * 0.6)
        x0 = max(0, cx - bw//2)
        y0 = max(0, cy - bh//2)
        crop = img_cv[y0:y0+bh, x0:x0+bw]
        cv2.imwrite(str(out_png), crop)
        return out_png

    img = Image.open(frame_path).convert("RGB")
    prompt = "Segment the main product and return an image part (transparent PNG) if available."
    resp = safe_generate_once(model, [prompt, img])
    if resp is None:
        print("[WARN] segmentation model call failed/empty; using deterministic crop fallback")
        img_cv = cv2.imread(str(frame_path))
        h, w = img_cv.shape[:2]
        cx, cy = w//2, h//2
        bw, bh = int(w * 0.6), int(h * 0.6)
        x0 = max(0, cx - bw//2)
        y0 = max(0, cy - bh//2)
        crop = img_cv[y0:y0+bh, x0:x0+bw]
        cv2.imwrite(str(out_png), crop)
        return out_png

    # Try to extract image bytes — if any step fails, fallback immediately
    try:
        parts = getattr(resp, "parts", None) or getattr(resp, "candidates", None)
        if parts:
            for part in parts:
                data = getattr(part, "data", None) or getattr(getattr(part, "content", None), "data", None)
                if isinstance(data, (bytes, bytearray)):
                    with open(out_png, "wb") as f:
                        f.write(data)
                    return out_png
                if getattr(part, "content", None) and getattr(part.content, "parts", None):
                    for sub in part.content.parts:
                        if getattr(sub, "mime_type", "").startswith("image/") and getattr(sub, "data", None):
                            with open(out_png, "wb") as f:
                                f.write(sub.data)
                            return out_png
    except Exception as e:
        print(f"[WARN] segmentation response parse error: {e}")

    print("[WARN] segmentation parse failed; using deterministic crop fallback")
    img_cv = cv2.imread(str(frame_path))
    h, w = img_cv.shape[:2]
    cx, cy = w//2, h//2
    bw, bh = int(w * 0.6), int(h * 0.6)
    x0 = max(0, cx - bw//2)
    y0 = max(0, cy - bh//2)
    crop = img_cv[y0:y0+bh, x0:x0+bw]
    cv2.imwrite(str(out_png), crop)
    return out_png

async def enhance_cropped_image(cropped_path: Path, run_id: str, styles: List[str]) -> List[Path]:
    results = []
    for i, style in enumerate(styles, start=1):
        out = OUTPUT_DIR / f"enh_{run_id}_{cropped_path.stem}_{i}.jpg"
        if not USE_REAL_GEMINI or genai is None:
            try:
                img = Image.open(cropped_path).convert("RGB")
                img = img.resize((1024, 1024))
                img.save(out, quality=90)
            except Exception:
                Image.new("RGB", (1024, 1024), (240,240,240)).save(out, quality=90)
            results.append(out)
            continue

        try:
            model = genai.GenerativeModel(GEMINI_IMAGE_MODEL)
        except Exception:
            print("[WARN] enhancement: couldn't init model; using fallback copy")
            img = Image.open(cropped_path).convert("RGB")
            img = img.resize((1024, 1024))
            img.save(out, quality=90)
            results.append(out)
            continue

        prompt = f"Create a professional high-res product photo of this object in style: {style}."
        img = Image.open(cropped_path).convert("RGB")
        resp = safe_generate_once(model, [prompt, img])
        if resp is None:
            print("[WARN] enhancement model call failed/empty; creating fallback copy")
            try:
                img = img.resize((1024, 1024))
                img.save(out, quality=90)
            except Exception:
                Image.new("RGB", (1024, 1024), (240,240,240)).save(out, quality=90)
            results.append(out)
            continue

        # Try to extract image bytes; fallback on any issue
        try:
            parts = getattr(resp, "parts", None) or getattr(resp, "candidates", None)
            saved = False
            if parts:
                for part in parts:
                    data = getattr(part, "data", None)
                    if isinstance(data, (bytes, bytearray)):
                        with open(out, "wb") as f:
                            f.write(data)
                        saved = True
                        break
                    if getattr(part, "content", None) and getattr(part.content, "parts", None):
                        for sub in part.content.parts:
                            if getattr(sub, "mime_type", "").startswith("image/") and getattr(sub, "data", None):
                                with open(out, "wb") as f:
                                    f.write(sub.data)
                                saved = True
                                break
                        if saved:
                            break
            if not saved:
                img_resize = img.resize((1024, 1024))
                img_resize.save(out, quality=90)
            results.append(out)
        except Exception as e:
            print(f"[WARN] enhancement response parse error: {e}. Falling back to local copy.")
            try:
                img = img.resize((1024, 1024))
                img.save(out, quality=90)
            except Exception:
                Image.new("RGB", (1024, 1024), (240,240,240)).save(out, quality=90)
            results.append(out)
    return results

# ---------------------------
# Top-level pipeline
# ---------------------------

async def run_pipeline(youtube_url: str, run_id: str, max_products: int = MAX_PRODUCTS) -> Dict[str, Any]:
    print(f"[INFO] RUN_ID={run_id} Starting pipeline for: {youtube_url}")
    video_path = download_youtube_video(youtube_url, run_id)
    print(f"[INFO] Video downloaded: {video_path}")

    frames = extract_three_frames_at_percentiles(video_path)
    print(f"[INFO] Extracted {len(frames)} frames: {[p.name for p in frames]}")

    detected_products: Dict[str, List[Path]] = {}
    for f in frames:
        dets = await identify_products_in_frame(f)
        dets = dets[:max_products]
        for d in dets:
            label = d.get("label", "product")
            detected_products.setdefault(label, []).append(f)

    products_out: Dict[str, Any] = {}
    processed = 0
    styles = ["studio white background", "lifestyle outdoors", "dramatic vignette"][:VARIATIONS]
    for label, frs in detected_products.items():
        if processed >= max_products:
            break
        try:
            best = await select_best_frame_for_product(frs, label)
            cropped = await segment_product_in_frame(best, run_id, bbox=None)
            enhanced = await enhance_cropped_image(cropped, run_id, styles)
            products_out[label] = {
                "best_frame": str(best),
                "cropped": str(cropped),
                "enhanced": [str(p) for p in enhanced],
            }
            processed += 1
        except Exception as e:
            print(f"[WARN] product pipeline failed for {label}: {e}")
            continue

    print(f"[INFO] RUN_ID={run_id} Pipeline complete. Processed products: {processed}")
    return {"youtube_url": youtube_url, "products": products_out}

# ---------------------------
# API endpoints
# ---------------------------

@app.post("/api/process")
async def process_video(req: ProcessRequest):
    try:
        reset_output_dir()
        run_id = uuid.uuid4().hex[:12]
        max_products = int(req.max_products) if req.max_products else MAX_PRODUCTS
        result = await run_pipeline(req.youtube_url, run_id, max_products=max_products)
        products = []
        for label, info in result.get("products", {}).items():
            products.append({
                "label": label,
                "best_frame": info["best_frame"],
                "cropped": info["cropped"],
                "enhanced": info["enhanced"],
            })
        return JSONResponse({"youtube_url": req.youtube_url, "products": products})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def serve_file(path: str, request: Request = None):
    try:
        file_path = Path(path)
        if file_path.is_absolute():
            resolved = file_path.resolve()
            if str(LG_WORKDIR) not in str(resolved):
                raise HTTPException(status_code=400, detail="Access outside workdir blocked")
            file_path = resolved
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        resp = FileResponse(str(file_path))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except HTTPException:
        raise
    except Exception as e:
        print(f"[WARN] serve_file error: {e}")
        raise HTTPException(status_code=500, detail="Unable to serve file")

@app.get("/api/health")
def health():
    return {"status": "ok"}
