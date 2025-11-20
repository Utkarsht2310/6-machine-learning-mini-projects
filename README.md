# 6 Machine Learning Mini Projects

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-js%20%7C%20tf.keras-ff6f00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

A grab bag of experiments that helped me practice end-to-end machine learning - from quick computer-vision prototypes to playful gesture-driven interfaces. Each folder is a self-contained mini project; together they showcase different workflows (classic notebooks, browser ML, and OpenCV pipelines).

## Projects at a Glance

| Project | Stack | What it Does | Key Learning |
| --- | --- | --- | --- |
| `air-juggler/` | TensorFlow.js, MediaPipe Hands, Canvas | Gesture-controlled browser game where you keep a ball aloft with virtual paddles mapped to your hands. | Real-time pose tracking latency, coordinate mirroring for mirrored webcams, and integrating ML inference inside a 60 FPS render loop. |
| `caption-for-image/` | InceptionV3 encoder + LSTM decoder | Notebook that preps Flickr8k captions, extracts CNN features, and trains a sequence model to generate captions. | Building dual-stream (vision + language) pipelines, cleaning noisy text, and balancing GPU memory by flattening feature embeddings. |
| `hand-filter/` | OpenCV, MediaPipe, NumPy | Applies different artistic filters exactly inside a quadrilateral defined by two thumbs + two index fingers. | Turning sparse landmarks into ROIs, blending filtered regions back onto frames, and designing intuitive gesture-driven UX cues. |
| `ipl-score-prediction/` | Pandas, Scikit-learn, Keras | Predicts IPL inning totals from live match context with a dense regression network plus a small widget for what-if analysis. | Encoding categorical cricket data, choosing Huber loss for stability, and exposing the model through lightweight ipywidgets. |
| `Number-plate-recognintion/` | (Notebook scaffold) | Workspace reserved for experimenting with OpenCV-based ANPR; currently focused on documenting preprocessing steps before coding. | Importance of dataset curation, understanding contour heuristics, and planning OCR-friendly pipelines before training. |

> NOTE: Want to tinker? Clone the repo, open the folder you care about, and follow the per-project instructions below.

## Getting Started

```bash
git clone https://github.com/Utkarsht2310/6-machine-learning-mini-projects.git
cd "6-machine-learning-mini-projects"
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt  # (Optional: install what each notebook needs)
```

> NOTE: Most notebooks expect you to provide datasets (Flickr8k images, IPL ball-by-ball CSV, etc.). See the notes below.

## Project Notes & Learnings

### 1. Air Juggler (`air-juggler/`)
- **What you get:** `index.html` boots a webcam-powered canvas, `handTracking.js` wires MediaPipe Hands through TensorFlow.js, and `game.js` drives physics, countdowns, and scoring overlays.
- **How to run:** Open `index.html` in a modern browser (HTTPS or `localhost` needed for camera access). Grant webcam permission.
- **Learning highlights:**
  - Synced the MediaPipe detector loop with the animation loop and throttled to ~30 FPS via `setTimeout` to keep inference stable.
  - Mirrored X coordinates (`x: 640 - avgX`) to match the mirrored camera feed and avoid control confusion.
  - Added loading + countdown overlays so the user sees when the detector is ready before physics kick in.

### 2. Image Captioning (`caption-for-image/`)
- **Notebook:** `caption-for-image/Untitled.ipynb`
- **Dataset:** Point `images_directory` and `captions_path` to your local Flickr8k download (Kaggle paths are used in-notebook for reference).
- **Learning highlights:**
  - Tokenization strategy wraps every caption with `start`/`end` tokens and stores paired `image_id\tcaption` strings to simplify splits.
  - Pre-computed InceptionV3 embeddings (flattened) dramatically accelerate caption training loops.
  - Visual diagnostics (`visualaization`, caption length histograms) caught outliers before training, saving GPU time.

### 3. Hand-Guided Filters (`hand-filter/hand_filter.py`)
- **Run it:** `pip install opencv-python mediapipe numpy` then `python hand_filter.py`.
- **Learning highlights:**
  - Used thumb + index finger landmarks on each hand to define a stable quadrilateral; `order_points` keeps corners consistent for masking.
  - Implemented multiple stylistic filters (grayscale, sepia, cartoon, etc.) and hot-swapped them by trapping key presses.
  - Blended filtered ROIs back into the original frame with masks so the effect feels AR-like instead of overlaying the whole frame.

### 4. IPL Score Prediction (`ipl-score-prediction/`)
- **Files:** `Untitled.ipynb`, `ipl_data.csv` (sample dataset).
- **Learning highlights:**
  - Encodes categorical cricket context via `LabelEncoder` and scales features with `MinMaxScaler` for dense networks.
  - Built a compact 3-layer `keras.Sequential` regressor trained with Huber loss to tame outliers in historical totals.
  - Exposed a mini ipywidgets UI where analysts can dial in venue/team combos and instantly compare predicted totals.

### 5. Number Plate Recognition Sandbox (`Number-plate-recognintion/`)
- **Status:** Notebook scaffold waiting for curated footage (why it currently has no executed cells).
- **Learning focus so far:**
  - Drafted the preprocessing plan: denoise -> morphological closing -> contour filtering before OCR.
  - Research showed that having clean, high-resolution regional datasets matters more than fancy architectures in early iterations.
  - Keeping a dedicated workspace ensures future experiments (EasyOCR vs. Tesseract, YOLO vs. Haar cascades) stay isolated.

## Suggested Next Steps

- Convert the notebooks into `.py` scripts or lightweight Streamlit apps for easier demos.
- Containerize the dependencies (especially for MediaPipe + GPU-heavy caption training).
- Flesh out the Number Plate notebook with OpenCV prototypes once data is ready.

---

If you spin up new experiments or improve any of these minis, feel free to open an issue or drop a PR!

