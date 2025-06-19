# 🧠 Deep Learning from fNIRS Time-Series  
*A Tutorial Companion for fNIRS classification tasks across two datasets and five deep learning models, including transformer and residual-nets*

This repository provides a hands-on implementation tutorial for the paper:  
**"Deep Learning from Diffuse Biomedical Optics Time-Series: An fNIRS-Focused Review of Recent Advancements and Future Directions."**

---

## 📂 Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Dataset Access](#2-dataset-access)
- [3. Preprocessing](#3-preprocessing)
- [4. Session-Wise Splitting (BSQ-HD)](#4-session-wise-splitting-bsq-hd)
- [5. Model Training](#5-model-training)
- [6. Saving and Logging](#6-saving-and-logging)
- [7. Citation](#7-citation)
- [8. Contact](#8-contact)

---

## 1. 🚀 Environment Setup

### Install Cedalion

Clone and install [Cedalion](https://github.com/ibs-lab/cedalion):

```bash
git clone https://github.com/ibs-lab/cedalion.git
cd cedalion
pip install -e .
```

### Install PyTorch

Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally) to install it with or without CUDA support.

---

## 2. 📥 Dataset Access

Please download the following datasets manually:

- **Ball Squeezing HD Dataset (BSQ-HD)**  
  Refer to the official publication for access and details.

- **Mental Arithmetic Multi-Modal Dataset**  
  Refer to the original publication for details.

---

## 3. 🧪 Preprocessing

Use `preprocessing.py` to convert raw recordings into model-ready datasets.

```bash
python preprocessing.py
```

- Edit `preprocessing.py` to set the correct dataset paths.
- By default, the dataset is filtered at **0.5 Hz**.
- To try additional frequency bands (used in the paper: `0.7 Hz`, `1.0 Hz`), modify the `FMAX` variable.

---

## 4. 📑 Session-Wise Splitting (BSQ-HD)

Run `sessions.py` to generate session-specific train/validation splits for BSQ-HD:

```bash
python sessions.py
```

This step is **only required for BSQ-HD**.

---

## 5. 🧠 Model Training

Each dataset folder includes multiple training scripts:

| Script             | Purpose                              |
|--------------------|---------------------------------------|
| `cvloso_e.py`      | Event-based classification            |
| `cvloso_ef.py`     | Event + Frequency model               |
| `cvloso_et.py`     | Event + Temporal structure            |
| `cvloso_eft.py`    | Full Data (Event + Frequency + Temporal) |

Update the training scripts with:
- Dataset file paths
- Any necessary config parameters (e.g., batch size, learning rate)

Make sure the required directories are created beforehand:
```bash
mkdir -p models loss
```

---

## 6. 💾 Saving and Logging

- Model checkpoints are saved per-epoch based on **validation loss**.
- Training loss, validation metrics, and learning curves are stored in the `loss/` directory.

Example output folder structure:

```
bsq_db/
├── data/
├── models/
├── loss/
├── preprocessing.py
├── sessions.py
└── cvloso_eft.py
```

---

## 7. 📚 Citation

If you use this code or find it helpful, please cite:

```
The paper is under review
```

---

## 8. 📬 Contact

For questions or collaborations, feel free to reach out:

- 🧑‍💻 Maintainer: [Your Name]
- 🏢 Affiliation: BIFOLD / TU Berlin, Monash University
- ✉️ Email: your.email@domain.com
