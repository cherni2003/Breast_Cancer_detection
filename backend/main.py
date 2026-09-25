"""
Mammo-CLIP FastAPI Backend — FIXED
"""

import os, sys

# ══════════════════════════════════════════════════════════════
#  CHEMINS — doit être EN PREMIER avant tout import local
# ══════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))   # backend/
ROOT = os.path.dirname(BASE)                         # mammograhie/

# Ajouter Mammo-CLIP au path AVANT d'importer inference_local
sys.path.insert(0, os.path.join(ROOT, 'Mammo-CLIP', 'src', 'codebase'))
sys.path.insert(0, BASE)  # pour trouver inference_local.py

from unittest.mock import MagicMock
sys.modules.setdefault('albumentations',                           MagicMock())
sys.modules.setdefault('albumentations.pytorch',                   MagicMock())
sys.modules.setdefault('albumentations.core',                      MagicMock())
sys.modules.setdefault('albumentations.core.transforms_interface', MagicMock())
sys.modules.setdefault('imgaug',                                   MagicMock())
sys.modules.setdefault('imgaug.augmenters',                        MagicMock())

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import uuid
import time

app = FastAPI(
    title="Mammo-CLIP API",
    description="Mammogram mass detection with GradCAM++",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = os.path.join(BASE, "uploads")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# ══════════════════════════════════════════════════════════════
#  Chargement du modèle
# ══════════════════════════════════════════════════════════════
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        try:
            from inference_local import load_classifier
            classifier = load_classifier()
            print("[INFO] Modele charge avec succes")
        except Exception as e:
            print(f"[ERROR] Impossible de charger le modele : {e}")
            import traceback
            traceback.print_exc()
            classifier = None
    return classifier

print("[INFO] Chargement du modele au demarrage...")
get_classifier()
print(f"[INFO] Modele charge : {classifier is not None}")


# ══════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Mammo-CLIP API is running",
        "model_loaded": classifier is not None
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.post("/analyze")
async def analyze_mammogram(file: UploadFile = File(...)):
    allowed = {".png", ".jpg", ".jpeg", ".dcm", ".tiff", ".tif"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Type {ext!r} non supporte. Utilise : {allowed}")

    job_id   = str(uuid.uuid4())[:8]
    img_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    start = time.time()
    clf   = get_classifier()

    if clf is None:
        raise HTTPException(503, "Modele non charge. Verifiez les checkpoints.")

    try:
        from inference_local import predict
        result = predict(
            img_path   = img_path,
            classifier = clf,
            img_size   = 512,
            output_dir = RESULTS_DIR,
            n_smooth   = 1,   # plus rapide pour l'API
        )
        prob = result["prob"]
        bbox = result["bbox_gradcam"]

        # Renommer le fichier output avec job_id
        default_out = result["output"]
        final_out   = os.path.join(RESULTS_DIR, f"gradcam_{job_id}.png")
        if os.path.exists(default_out) and default_out != final_out:
            shutil.move(default_out, final_out)

        heatmap = f"/results/gradcam_{job_id}.png"

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur inference : {e}")

    elapsed = round(time.time() - start, 2)

    return JSONResponse({
        "job_id"     : job_id,
        "filename"   : file.filename,
        "prob"       : round(float(prob), 4),
        "prediction" : "MASSE" if prob >= 0.5 else "Normal",
        "confidence" : (
            "HIGH"   if abs(prob - 0.5) > 0.25 else
            "MEDIUM" if abs(prob - 0.5) > 0.10 else
            "LOW"
        ),
        "bbox"       : bbox,
        "heatmap_url": heatmap,
        "elapsed_s"  : elapsed,
        "demo_mode"  : False,
    })


@app.get("/result/{job_id}")
def get_result(job_id: str):
    out = os.path.join(RESULTS_DIR, f"gradcam_{job_id}.png")
    if not os.path.exists(out):
        raise HTTPException(404, "Result not found")
    return FileResponse(out, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)