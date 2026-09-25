# Breast Cancer Detection from Mammography Images

An AI-powered system for automated breast cancer detection, classification, and lesion localization from mammography images, combining multiple deep learning approaches with visual interpretability (Grad-CAM).

## 🔍 Overview

This project explores and combines **three complementary modeling approaches** for breast cancer analysis on mammograms:

1. **Direct CNN Classification** — an EfficientNet-B5 model fine-tuned end-to-end for benign/malignant classification.
2. **CLIP-based Embeddings + Linear Probe** — feature embeddings extracted with **Mammo-CLIP** (a mammography-specialized foundation model), classified using a lightweight linear probe (StandardScaler + Logistic Regression).
3. **Lesion Detection** — a **RetinaNet**-based object detector for localizing suspicious regions on mammograms.

All predictions are paired with **Grad-CAM++** heatmaps to visually explain which regions of the image influenced the model's decision. The whole pipeline is exposed through a **React web app** talking to a **FastAPI backend**.

## ✨ Features

- 🩻 Mammogram classification via EfficientNet-B5 CNN
- 🧬 Embedding-based classification via Mammo-CLIP + linear probe
- 🎯 Lesion localization via RetinaNet
- 🔥 Grad-CAM++ visual explanations for every prediction
- 📤 Drag-and-drop web interface for uploading mammograms (PNG, JPG, TIFF, **DICOM**)
- ☁️ Trained model weights hosted on Hugging Face (see below)

## 🤗 Models

The trained model weights are **not stored in this GitHub repository** (they are too large for standard Git). They are hosted on Hugging Face:

👉 **https://huggingface.co/Oumaima2003/Breast_cancer_model**

This includes:
- `b5-model-best-epoch-7.tar` and `best_model_auc0796_recall0853.pth` — EfficientNet-B5 classifier checkpoints (AUC 0.796, Recall 0.853)
- `linear_probe_models.pkl` — Logistic Regression + StandardScaler trained on Mammo-CLIP embeddings
- Mammo-CLIP and RetinaNet related weights/configs

To use the models locally, download them from the Hugging Face repo (Git LFS/Xet required):

```bash
git lfs install
git clone https://huggingface.co/Oumaima2003/Breast_cancer_model
```

Or download individual files directly from the [Files tab](https://huggingface.co/Oumaima2003/Breast_cancer_model/tree/main).

## 🏗️ Project Structure

```
.
├── Breast_cancer_model/     # EfficientNet-B5 checkpoints (.tar, .pth)
├── Mammo-CLIP/               # Mammo-CLIP model / embeddings pipeline
├── backend/                  # FastAPI inference service
│   ├── main.py                 # API entrypoint (FastAPI + uvicorn)
│   ├── inference_local.py       # Model loading & prediction logic
│   ├── requirements.txt
│   ├── uploads/                 # User-uploaded mammography images
│   ├── results/                  # Predictions & Grad-CAM overlays
│   └── models_export/
├── frontend/                  # React (Vite) web interface
│   ├── src/
│   │   ├── App.jsx               # Main UI (upload, results, heatmap display)
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models_export/            # Exported/converted model artifacts
├── outputs/                    # Generated outputs
├── linear_probe_models.pkl    # Linear probe (StandardScaler + LogisticRegression)
├── Mammo retinanet inference.py   # RetinaNet lesion detection script
├── gradcam_local.py            # Grad-CAM generation script
├── .gitattributes             # Git LFS tracking rules
└── README.md
```

## 🧠 Models Summary

| Approach | Model | Metric(s) |
|---|---|---|
| Direct classification | EfficientNet-B5 | AUC 0.796, Recall 0.853 |
| Embedding + linear probe | Mammo-CLIP + Logistic Regression | *[add metric]* |
| Lesion detection | RetinaNet | *[add metric]* |

## 🖥️ Frontend

Built with **React 18** and **Vite**.

**Key dependencies:**
- `react-dropzone` — drag-and-drop mammogram upload
- `framer-motion` — UI animations
- `lucide-react` — icon set

**Setup & run:**
```bash
cd frontend
npm install
npm run dev      # starts on http://localhost:3000
```

Build for production:
```bash
npm run build
npm run preview
```

The frontend lets the user upload a mammography image, sends it to the backend `/analyze` endpoint, and displays the classification result along with the Grad-CAM++ heatmap returned by the API.

## ⚙️ Backend

Built with **FastAPI** (served via `uvicorn`), exposing a REST API for inference. On startup, it loads the EfficientNet-B5 classifier (`inference_local.load_classifier()`) and integrates the **Mammo-CLIP** codebase for feature extraction.

**Setup & run:**
```bash
cd backend
pip install -r requirements.txt
python main.py      # starts on http://localhost:8000
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status & whether the model is loaded |
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Upload a mammogram (`.png`, `.jpg`, `.jpeg`, `.dcm`, `.tiff`, `.tif`) and get a prediction |
| `GET` | `/result/{job_id}` | Retrieve the generated Grad-CAM heatmap image |

**Example `/analyze` response:**
```json
{
  "job_id": "a1b2c3d4",
  "filename": "mammogram.png",
  "prob": 0.8123,
  "prediction": "MASSE",
  "confidence": "HIGH",
  "bbox": [...],
  "heatmap_url": "/results/gradcam_a1b2c3d4.png",
  "elapsed_s": 1.42,
  "demo_mode": false
}
```

Each uploaded image runs through the EfficientNet-B5 classifier at 512×512 resolution, and a Grad-CAM++ heatmap plus bounding box are generated and saved to `backend/results/`, then served back to the frontend.

## 📈 Results

Sample outputs (classification results and Grad-CAM overlays) are available in `backend/results/` and `outputs/`.

## 🎓 Context

Developed as part of an engineering curriculum in Data Science & Artificial Intelligence.

## 📄 License

*[Add your license, e.g. MIT]*

## 🙋 Author

**Oumaima Cherni (Mimi)**
Engineering student — Data Science & Artificial Intelligence, TEK-UP
