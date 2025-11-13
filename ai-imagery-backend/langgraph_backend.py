"""
Pure LangGraph backend compatible with your installed version.
SYNC graph (no await). START node entry. Full pipeline: download → frames → identify → select → segment → enhance → collect.
"""

import os
import uuid
import shutil
import json
import subprocess
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import cv2
from dotenv import load_dotenv

# ---- LangGraph imports (your version) ----
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------
# Config
# ---------------------------
USE_REAL_GEMINI = os.environ.get("USE_REAL_GEMINI", "false").lower() in ("1","true","yes")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)

MAX_PRODUCTS = int(os.environ.get("MAX_PRODUCTS", "2"))
VARIATIONS   = int(os.environ.get("VARIATIONS", "3"))
LG_WORKDIR   = Path(os.environ.get("LG_WORKDIR", "./workdir")).absolute()

VIDEO_DIR  = LG_WORKDIR / "videos"
FRAMES_DIR = LG_WORKDIR / "frames"
OUTPUT_DIR = LG_WORKDIR / "output"
for d in (VIDEO_DIR, FRAMES_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Gemini init
# ---------------------------
genai = None
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
    except:
        genai = None

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="LangGraph Backend (SYNC)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    youtube_url: str
    max_products: Optional[int] = MAX_PRODUCTS

# ---------------------------
# LangGraph State Schema
# ---------------------------
class PipelineState(TypedDict, total=False):
    inputs: Dict[str, Any]
    data: Dict[str, Any]
    outputs: Dict[str, Any]

graph = StateGraph(PipelineState)

# ---------------------------
# Utility
# ---------------------------
def reset_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def youtube_download(url: str, run_id: str) -> str:
    out_path = VIDEO_DIR / f"video_{run_id}.mp4"
    safe_mkdir(VIDEO_DIR)

    if out_path.exists():
        return str(out_path)

    if shutil.which("yt-dlp"):
        subprocess.check_call([
            "yt-dlp",
            "-f", "best[ext=mp4]/mp4",
            "-o", str(out_path),
            url
        ])
        return str(out_path)

    from pytube import YouTube
    yt = YouTube(url)
    stream = yt.streams.filter(
        progressive=True, file_extension='mp4'
    ).order_by('resolution').desc().first()
    stream.download(
        output_path=str(VIDEO_DIR),
        filename=out_path.name
    )
    return str(out_path)

def extract_frames(video_path: str) -> List[str]:
    frames_dir = FRAMES_DIR / Path(video_path).stem
    safe_mkdir(frames_dir)

    # reuse if exists
    found = sorted(frames_dir.glob("frame_*.jpg"))
    if len(found) >= 3:
        return [str(p) for p in found[:3]]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25
    duration    = frame_count/fps if fps>0 else 0

    saved=[]
    if duration <= 0:
        # fallback
        while len(saved)<3:
            ret, frame = cap.read()
            if not ret: break
            out = frames_dir / f"frame_{len(saved):06d}.jpg"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out)
            saved.append(str(out))
        return saved

    for i,p in enumerate([0.2,0.5,0.8]):
        t = duration * p
        frame_no = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret: continue
        out = frames_dir / f"frame_{i:06d}.jpg"
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out)
        saved.append(str(out))

    return saved

# ---------------------------
# Gemini helpers
# ---------------------------
def safe_genai_once(model, inp):
    if model is None: 
        return None
    try:
        if hasattr(model,"generate_content"):
            return model.generate_content(inp)
        if hasattr(model,"predict"):
            return model.predict(inp)
        return None
    except Exception as e:
        print("Gemini failed:", e)
        return None

def extract_text(resp) -> str:
    if resp is None: return ""
    try:
        return resp.text or ""
    except:
        return ""

# ---------------------------
# Nodes
# ---------------------------
def n_download(state: PipelineState) -> PipelineState:
    url = state["inputs"]["youtube_url"]
    run = state["inputs"]["run_id"]
    vp  = youtube_download(url, run)
    state.setdefault("data",{})["video_path"] = vp
    return state

def n_frames(state: PipelineState) -> PipelineState:
    vp = state["data"]["video_path"]
    state["data"]["frames"] = extract_frames(vp)
    return state

def n_identify(state: PipelineState) -> PipelineState:
    frames = state["data"]["frames"]
    detected: Dict[str,List[str]] = {}

    for f in frames:
        labels=None
        if genai:
            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                img   = Image.open(f).convert("RGB")
                resp  = safe_genai_once(model, ["Identify JSON", img])
                txt   = extract_text(resp)
                if "[" in txt and "]" in txt:
                    arr = json.loads(txt[txt.find("["):txt.rfind("]")+1])
                    labels = [p.get("label","product") for p in arr]
            except:
                labels=None

        if not labels:
            labels=["product"]

        labels = labels[:state["inputs"]["max_products"]]
        for lab in labels:
            detected.setdefault(lab,[]).append(f)

    state["data"]["detected_products"]=detected
    return state

def n_select(state: PipelineState) -> PipelineState:
    det = state["data"]["detected_products"]
    selected={}

    for lab,frs in det.items():
        best = frs[0]
        if genai and len(frs)>1:
            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                imgs = [Image.open(f).convert("RGB") for f in frs]
                resp = safe_genai_once(model, [f"pick for {lab}", *imgs])
                txt  = extract_text(resp)
                for f in frs:
                    if Path(f).name in txt:
                        best = f
                        break
            except:
                pass
        selected[lab] = best

    state["data"]["selected_frames"] = selected
    return state

def n_segment(state: PipelineState) -> PipelineState:
    sel = state["data"]["selected_frames"]
    run = state["inputs"]["run_id"]
    cropped={}

    for lab,fp in sel.items():
        out = OUTPUT_DIR / f"seg_{run}_{Path(fp).stem}_{lab}.png"
        success=False

        if genai:
            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                img   = Image.open(fp).convert("RGB")
                resp  = safe_genai_once(model, ["segment", img])
                parts = getattr(resp,"parts",None)
                if parts:
                    for p in parts:
                        data = getattr(p,"data",None)
                        if isinstance(data,(bytes,bytearray)):
                            out.write_bytes(data)
                            success=True
                            break
            except:
                success=False

        if not success:
            im = cv2.imread(fp)
            if im is None:
                Image.new("RGB",(512,512),(200,200,200)).save(out)
            else:
                h,w = im.shape[:2]
                cx,cy = w//2, h//2
                bw,bh = int(w*0.6), int(h*0.6)
                x0=max(0,cx-bw//2); y0=max(0,cy-bh//2)
                crop = im[y0:y0+bh, x0:x0+bw]
                cv2.imwrite(str(out), crop)

        cropped[lab]=str(out)

    state["data"]["cropped_paths"]=cropped
    return state

def n_enhance(state: PipelineState) -> PipelineState:
    """
    Ultra-simple enhancement:
    Generates 3 REAL variations:
    1. White background
    2. Light gray background + resized
    3. Light blue background + centered
    No blur, no filters, no rotations → zero errors.
    """
    from PIL import Image

    cropped = state["data"]["cropped_paths"]
    run     = state["inputs"]["run_id"]
    enhanced={}

    for label, fp in cropped.items():
        base = Image.open(fp).convert("RGBA")
        w, h = base.size

        outputs=[]

        # ─────────────────────────────────────────
        # 1️⃣ White background (default)
        # ─────────────────────────────────────────
        out1 = OUTPUT_DIR / f"enh_{run}_{label}_1.png"
        bg1 = Image.new("RGBA", (w,h), (255,255,255,255))
        bg1.paste(base, (0,0), base)
        bg1.convert("RGB").save(out1, quality=95)
        outputs.append(str(out1))

        # ─────────────────────────────────────────
        # 2️⃣ Light gray + smaller product (0.9x)
        # ─────────────────────────────────────────
        out2 = OUTPUT_DIR / f"enh_{run}_{label}_2.png"
        bg2 = Image.new("RGBA", (w,h), (230,230,230,255))

        scaled = base.resize((int(w*0.9), int(h*0.9)))
        sx = (w - scaled.width)//2
        sy = (h - scaled.height)//2
        bg2.paste(scaled, (sx,sy), scaled)

        bg2.convert("RGB").save(out2, quality=95)
        outputs.append(str(out2))

        # ─────────────────────────────────────────
        # 3️⃣ Light blue + even smaller product (0.8x)
        # ─────────────────────────────────────────
        out3 = OUTPUT_DIR / f"enh_{run}_{label}_3.png"
        bg3 = Image.new("RGBA", (w,h), (200,220,255,255))

        scaled2 = base.resize((int(w*0.8), int(h*0.8)))
        sx2 = (w - scaled2.width)//2
        sy2 = (h - scaled2.height)//2
        bg3.paste(scaled2, (sx2,sy2), scaled2)

        bg3.convert("RGB").save(out3, quality=95)
        outputs.append(str(out3))

        enhanced[label] = outputs

    state["data"]["enhanced"] = enhanced
    return state

def n_collect(state: PipelineState) -> PipelineState:
    det=state["data"]["detected_products"]
    sel=state["data"]["selected_frames"]
    crop=state["data"]["cropped_paths"]
    enh =state["data"]["enhanced"]

    products=[]
    for lab in det.keys():
        products.append({
            "label": lab,
            "best_frame": sel.get(lab),
            "cropped": crop.get(lab),
            "enhanced": enh.get(lab,[])
        })

    state["outputs"]={
        "youtube_url": state["inputs"]["youtube_url"],
        "run_id":      state["inputs"]["run_id"],
        "products":    products
    }
    return state

# ---------------------------
# Add nodes
# ---------------------------
graph.add_node("download_video", n_download)
graph.add_node("extract_frames", n_frames)
graph.add_node("identify_products", n_identify)
graph.add_node("select_frames", n_select)
graph.add_node("segment_products", n_segment)
graph.add_node("enhance_products", n_enhance)
graph.add_node("collect_results", n_collect)

# ---------------------------
# START node
# ---------------------------
from langgraph.graph import START
graph.add_edge(START, "download_video")

# ---------------------------
# Edges
# ---------------------------
graph.add_edge("download_video", "extract_frames")
graph.add_edge("extract_frames", "identify_products")
graph.add_edge("identify_products", "select_frames")
graph.add_edge("select_frames", "segment_products")
graph.add_edge("segment_products", "enhance_products")
graph.add_edge("enhance_products", "collect_results")
graph.add_edge("collect_results", END)

# ---------------------------
# Compile graph
# ---------------------------
app_graph = graph.compile()

# ---------------------------
# SYNC runner (IMPORTANT)
# ---------------------------
def run_graph(state: PipelineState):
    if hasattr(app_graph, "invoke"):
        return app_graph.invoke(state)
    if hasattr(app_graph, "run"):
        return app_graph.run(state)
    if hasattr(app_graph, "execute"):
        return app_graph.execute(state)
    raise RuntimeError("No valid graph executor found")

# ---------------------------
# FastAPI
# ---------------------------
@app.post("/api/process")
async def process_video(req: ProcessRequest):
    try:
        reset_output_dir()
        run_id = uuid.uuid4().hex[:12]

        init_state:PipelineState = {
            "inputs":{
                "youtube_url": req.youtube_url,
                "run_id": run_id,
                "max_products": int(req.max_products or MAX_PRODUCTS),
            },
            "data":{},
            "outputs":{}
        }

        result = run_graph(init_state)
        return JSONResponse(result.get("outputs", {}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status":"ok"}


@app.get("/files")
async def files(path:str):
    f = Path(path)
    if f.is_absolute() and LG_WORKDIR not in f.resolve().parents:
        raise HTTPException(status_code=400, detail="outside workdir")
    if not f.exists():
        raise HTTPException(status_code=404, detail="not found")
    resp = FileResponse(str(f))
    resp.headers["Cache-Control"] = "no-store"
    return resp
