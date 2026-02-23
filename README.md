
# RAG – Retrieval-Augmented Generation Pipeline

This repository contains the full Retrieval-Augmented Generation (RAG) experimental pipeline
with configuration-driven experiments.

Repository structure:

RAG/
├── src/        # All source scripts
├── configs/    # YAML experiment configurations
├── data/       # Dataset (download from Zenodo)
└── README.md

Dataset is hosted on Zenodo and should be downloaded separately
to avoid GitHub large file limits.

------------------------------------------------------------
🚀 QUICK START (RUN EVERYTHING STEP-BY-STEP)
------------------------------------------------------------

1) Clone Repository

git clone https://github.com/projectanonymous456/RAG.git
cd RAG


2) Create Virtual Environment

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

If requirements.txt is not available:

pip install torch transformers datasets faiss-cpu sentence-transformers pyyaml scikit-learn


3) Download Dataset from Zenodo

Download dataset manually and place it in:

data/raw/

Expected structure:

RAG/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── configs/
└── README.md


------------------------------------------------------------
🧪 FULL PIPELINE COMMANDS
------------------------------------------------------------

# 1. (Optional) Preprocess dataset
python src/01_preprocess.py --config configs/mozilla.yaml

# 2. Generate synthetic data (M2 - prompt only)
python src/05_generate_aug.py     --config configs/mozilla.yaml     --workdir workdir     --mode m2

# 3. Generate synthetic data (M3 - RAG based, K=5)
python src/05_generate_aug.py     --config configs/mozilla.yaml     --workdir workdir     --mode m3     --k 5     --m3_st_device cpu

# 4. Train & Evaluate Baseline (M1)
python src/06_train_eval.py     --config configs/mozilla.yaml     --workdir workdir     --mode m1

# 5. Train & Evaluate with M2
python src/06_train_eval.py     --config configs/mozilla.yaml     --workdir workdir     --mode m2

# 6. Train & Evaluate with M3
python src/06_train_eval.py     --config configs/mozilla.yaml     --workdir workdir     --mode m3


------------------------------------------------------------
📊 OUTPUTS
------------------------------------------------------------

All outputs are saved under:

workdir/outputs/<dataset>/

Includes:
- Synthetic JSONL files
- Logs
- Metrics JSON files
- Summary results

Example:

workdir/outputs/mozilla/metrics/m3_r10_k5_ALL.json


------------------------------------------------------------
⚡ RUN SINGLE RATIO ONLY (OPTIONAL)
------------------------------------------------------------

python src/05_generate_aug.py     --config configs/mozilla.yaml     --workdir workdir     --mode m3     --k 5     --ratios r10


------------------------------------------------------------
🖥 GOOGLE COLAB VERSION
------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive')

!python /content/drive/MyDrive/RAG/src/05_generate_aug.py     --config /content/drive/MyDrive/RAG/configs/mozilla.yaml     --workdir /content/drive/MyDrive/RAG/workdir     --mode m3 --k 5


------------------------------------------------------------
🔧 TROUBLESHOOTING
------------------------------------------------------------

CUDA Out Of Memory:
- Use --m3_st_device cpu
- Or enable 4-bit quantization if supported.

FAISS error:
pip install faiss-cpu


------------------------------------------------------------
📜 CITATION
------------------------------------------------------------

If you use this code, please cite:
- Zenodo Dataset DOI
- This GitHub repository
