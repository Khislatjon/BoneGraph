# Phase 3 — VLM Integration Plan

**Status: ⏳ Planned**

Add image understanding for X-ray and MRI inputs. The end result: a user uploads a bone image → the system describes the findings → retrieves relevant literature → the LLM gives a grounded, evidence-backed answer.

---

## Architecture

```
Image upload (X-ray / MRI)
        │
        ▼
   LLaVA 1.6 (VLM)
   generates structured radiological report
        │
        ├──► Embed report with SPECTER2 (adhoc_query adapter)
        │              │
        │              ▼
        │    Cosine search over chunks.db
        │    (same retrieval as text queries)
        │              │
        └──► Report + retrieved passages + user question
                       │
                       ▼
              HuatuoGPT-o1-8B
              streams grounded answer
```

**Key architectural decision:** VLM reports are embedded with SPECTER2 — the same model used for text chunks. This puts visual findings and literature in the same vector space, enabling cross-modal retrieval without a separate bridge model. A visual finding like "cortical thinning at femoral neck" automatically retrieves literature on cortical thinning mechanics because SPECTER2 encodes semantics, not modality.

---

## Step 1 — Collect image datasets

### Overview

| Dataset | Modality | Content | Images | Disk | Priority |
|---|---|---|---|---|---|
| **MURA** | X-ray | 7 upper extremity parts · normal/abnormal | 40,561 | ~6 GB | ⭐ Start here |
| **RSNA Bone Age** | Hand X-ray | Paediatric bone age · Greulich-Pyle graded | 12,611 | ~4 GB | ⭐ Start here |
| **OAI** | X-ray + MRI | Knee osteoarthritis · longitudinal | ~10M images | Very large | Later |
| **VerSe** | CT | Vertebral labelling + segmentation | ~300 scans | ~15 GB | Later |
| **Osteoporosis X-ray** | Spine/hip X-ray | Osteoporosis graded | ~4,800 | ~1 GB | Later |
| **TCIA — TCGA-SARC** | MRI / CT | Bone sarcoma | ~2,200 series | ~50 GB | Later |

---

### Dataset 1 — MURA (Stanford)

**What it contains:** 40,561 X-rays of 7 upper extremity body parts — shoulder, clavicle, elbow, finger, forearm, humerus, wrist. Each study is labelled normal or abnormal by radiologists.

**Why useful for BoneMind:** Covers cortical bone abnormalities, fractures, and structural changes across multiple bone types with radiologist-validated labels.

**How to get it:**

1. Go to the Stanford ML Group MURA page (search "Stanford MURA dataset")
2. Fill in the short registration form (name, institution, intended use) — approval is immediate
3. You receive a download link by email, typically within minutes
4. Download the archive (~6 GB zip) and extract it

**File structure after extraction:**
```
MURA-v1.1/
├── train/
│   ├── XR_SHOULDER/
│   │   ├── patient00001/
│   │   │   └── study1_positive/     ← abnormal
│   │   │       ├── image1.png
│   │   │       └── image2.png
│   │   └── patient00002/
│   │       └── study1_negative/     ← normal
│   ├── XR_ELBOW/
│   ├── XR_FINGER/
│   ├── XR_FOREARM/
│   ├── XR_HUMERUS/
│   ├── XR_WRIST/
│   └── XR_HAND/
├── valid/
│   └── ... (same structure)
├── train_labeled_studies.csv        ← path, label (0=normal, 1=abnormal)
└── valid_labeled_studies.csv
```

**Labels:** The CSV files map each study folder to a binary label. `study1_positive` = abnormal, `study1_negative` = normal. The folder name is the ground truth.

**Place in BoneMind:**
```
data/raw/images/MURA/train/XR_SHOULDER/patient00001/...
data/raw/images/MURA/valid/XR_SHOULDER/...
```

---

### Dataset 2 — RSNA Bone Age Challenge

**What it contains:** 12,611 hand X-rays of children aged 1–18, each labelled with bone age in months by radiologists. Sourced from the Radiological Society of North America 2017 challenge.

**Why useful for BoneMind:** Hand X-rays show cortical thickness, bone density, growth plate status, and skeletal maturity — all relevant to bone morphology understanding.

**How to get it:**

1. Create a free Kaggle account if you don't have one (kaggle.com)
2. Search for "RSNA Bone Age" on Kaggle — the competition dataset is publicly available
3. Accept the competition rules (one click)
4. Download via the Kaggle web UI, or install the Kaggle CLI:

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle competitions download -c rsna-bone-age
unzip rsna-bone-age.zip -d data/raw/images/RSNA_BoneAge/
```

**File structure after extraction:**
```
RSNA_BoneAge/
├── boneage-training-dataset/
│   ├── 1377.png
│   ├── 1378.png
│   └── ...                          ← filenames are image IDs
├── boneage-test-dataset/
│   └── ...
├── train.csv                        ← id, boneage (months), male (bool)
└── test.csv                         ← id only (labels withheld for competition)
```

**Labels:** `train.csv` has `id`, `boneage` (in months), and `male` (sex). Bone age in months is continuous — you can bin it: <120 months = child, 120–216 = adolescent.

**Place in BoneMind:**
```
data/raw/images/RSNA_BoneAge/
```

---

### Dataset 3 — OAI (NIH Osteoarthritis Initiative) — for later

**What it contains:** Longitudinal study of 4,796 subjects tracked over 8 years. Includes bilateral knee X-rays (posteroanterior and lateral), knee MRI (3T), and DXA scans at multiple time points.

**Why useful for BoneMind:** Best available public dataset for osteoarthritis progression, knee bone structure, and cartilage loss over time. KL grades (0–4) are provided for osteoarthritis severity.

**How to get it:**

1. Go to nda.nih.gov and create an account
2. Search for "Osteoarthritis Initiative" — the dataset is under the NIMH Data Archive
3. Submit a data access request — requires brief description of research use; typically approved within 1–2 weeks
4. Download is managed through the NDA Download Manager (a Java tool they provide)

**Important:** The full OAI dataset is very large (~several TB for all imaging). Download selectively:
- Baseline knee X-rays only: ~20 GB, manageable
- Baseline knee MRI: ~500 GB per time point — only download if you specifically need MRI

**Labels available:** KL grade per knee per visit, BMI, age, sex, pain scores.

---

### Dataset 4 — VerSe (Vertebral Segmentation) — for later

**What it contains:** 374 CT scans with manually annotated vertebral labels (C1–L5) and segmentation masks. From a Grand Challenge competition.

**Why useful for BoneMind:** Spine CT covers vertebral morphology, fracture detection, and bone density estimation — important for osteoporosis and spinal pathology.

**How to get it:**

1. Search "VerSe 2020 Grand Challenge" — the dataset is hosted on Zenodo and GitHub
2. No registration required — direct download from Zenodo
3. Files are in NIfTI format (`.nii.gz`) — requires nibabel or SimpleITK to read

```bash
pip install nibabel
```

**File format note:** CT NIfTI files are 3D volumes, not 2D images. You will need a slice extraction step to convert 3D CT volumes into 2D images for LLaVA:

```python
import nibabel as nib
import numpy as np
from PIL import Image

img = nib.load("verse001.nii.gz")
volume = img.get_fdata()
# Extract axial slices
for i in range(volume.shape[2]):
    slice_2d = volume[:, :, i]
    # Normalise to 0–255
    slice_norm = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)
    Image.fromarray(slice_norm).save(f"verse001_slice_{i:03d}.png")
```

---

### Dataset 5 — Osteoporosis X-ray (Kaggle) — for later

**What it contains:** Spine and hip X-rays graded for osteoporosis (normal, osteopenia, osteoporosis). Several versions exist on Kaggle under different competition names.

**How to get it:**

1. Search Kaggle for "osteoporosis x-ray classification"
2. The most used version has ~4,800 images in 3 classes
3. Download via Kaggle CLI:

```bash
kaggle datasets download -d <dataset-slug>   # check exact slug on Kaggle
unzip <file>.zip -d data/raw/images/Osteoporosis_Xray/
```

---

### Dataset 6 — TCIA (TCGA-SARC bone sarcoma) — for later

**What it contains:** MRI and CT scans of bone sarcoma patients from The Cancer Imaging Archive. Includes pre- and post-treatment scans.

**How to get it:**

1. Go to cancerimagingarchive.net
2. Search for "TCGA-SARC"
3. Create a free account — no institutional approval needed
4. Download via the TCIA Data Retriever desktop app (they provide it), or via their REST API

**File format:** DICOM (`.dcm`). You will need `pydicom` to read them:

```bash
pip install pydicom
```

```python
import pydicom
import numpy as np
from PIL import Image

ds = pydicom.dcmread("CT000001.dcm")
pixel_array = ds.pixel_array.astype(float)
pixel_norm = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
Image.fromarray(pixel_norm).save("CT000001.png")
```

---

### Where to place all datasets

```
data/raw/images/
├── MURA/
│   ├── train/
│   └── valid/
├── RSNA_BoneAge/
│   ├── boneage-training-dataset/
│   └── boneage-test-dataset/
├── OAI/                             ← add later
├── VerSe/                           ← add later
├── Osteoporosis_Xray/               ← add later
└── TCIA_SARC/                       ← add later
```

All image data is gitignored — add to `.gitignore`:
```
data/raw/images/
```

---

### What to capture per image (for images.db)

| Field | MURA | RSNA Bone Age | OAI | VerSe |
|---|---|---|---|---|
| `modality` | xray | xray | xray / mri | ct |
| `body_part` | from folder name (XR_WRIST etc.) | hand | knee | spine |
| `label` | normal / abnormal | bone_age_months | KL grade 0–4 | vertebra level |
| `subject_id` | patient folder name | image ID from CSV | subject ID | scan filename |

---

## Step 2 — Image database and ingestion pipeline

### Directory layout

```
data/raw/images/<dataset>/<subject_id>/<image.png>
data/db/images.db
```

### images.db schema

```sql
CREATE TABLE images (
    image_id         TEXT PRIMARY KEY,
    file_path        TEXT NOT NULL,
    modality         TEXT,           -- xray, mri, ct
    body_part        TEXT,           -- wrist, knee, spine, hip, hand
    source           TEXT,           -- MURA, OAI, RSNA_BoneAge, VerSe
    label            TEXT,           -- normal, abnormal, osteoporosis, fracture
    vlm_report       TEXT,           -- filled by describe_images.py
    report_embedding BLOB,           -- SPECTER2 embedding of vlm_report
    created_at       TEXT
);
```

### New files

```
ingestion/images/
    pipeline.py     # scans dataset folders, reads label CSVs, registers in images.db
    storage.py      # SQLite read/write helpers for images.db
```

---

## Step 3 — VLM setup

### Model: LLaVA 1.6 via Ollama

```bash
ollama pull llava:34b   # best quality (~20 GB)
ollama pull llava:13b   # good quality, faster (~8 GB)
```

LLaVA 1.6 runs locally via Ollama — no API costs, no data leaving the machine, and it significantly outperforms 1.5 on medical images without any fine-tuning.

### Bone-specific VLM prompt

```
You are a radiologist specialised in bone and musculoskeletal imaging.
Describe this image systematically:
1. Modality and body part
2. Cortical bone: thickness, continuity, any thinning or erosion
3. Trabecular bone: density, pattern, any rarefaction
4. Joint spaces if visible
5. Any fractures, lesions, or pathological changes
6. Overall impression in 1–2 sentences
Use precise radiological terminology.
```

### Phase 3b — fine-tuning (novel contribution for the paper)
After the base pipeline is working, fine-tune LLaVA on the collected bone image dataset using LoRA. This produces more precise bone-specific descriptions and is the key novelty claim for the VLM component.

---

## Step 4 — Image description pipeline

Create `processing/describe_images.py`:

1. Load images from `images.db` where `vlm_report IS NULL`
2. Send each image to LLaVA via Ollama with the bone-specific prompt
3. Store the report in `images.db`
4. Embed the report with SPECTER2 → store as `report_embedding`

```bash
python -m processing.describe_images          # process all unprocessed images
python -m processing.describe_images --limit 100   # test on first 100
```

---

## Step 5 — Cross-modal retrieval

Extend `retrieval/retriever.py` to accept image reports as queries:

```python
# Existing text query path
results = retriever.query("cortical bone fracture toughness", top_k=8)

# New image query path — report is already text, uses same method
results = retriever.query(vlm_report, top_k=8)
```

No changes needed to the core retrieval logic — SPECTER2 handles both. The only addition is a `describe_image(image_path)` function that calls LLaVA and returns the report string.

---

## Step 6 — Gradio UI — new "Analyse Image" tab

Add a third tab to `app.py`:

```
┌──────────────────────────────────────────────────────────────┐
│  Upload X-ray or MRI          Optional question              │
│  [Drop image here]            [What are the fracture risk…]  │
│                                              [🔬 Analyse]    │
├───────────────────────┬──────────────────────────────────────┤
│  VLM Report           │  Retrieved passages                  │
│  (LLaVA findings)     │  (from text corpus)                  │
├───────────────────────┴──────────────────────────────────────┤
│  Answer (HuatuoGPT — grounded in report + literature)        │
└──────────────────────────────────────────────────────────────┘
```

New Gradio components needed:
- `gr.Image(type="filepath")` for image upload
- `gr.Textbox` for optional question (default: "Describe the findings and their clinical significance")
- Reuse existing `answer_md` and `sources_html` pattern from Tab 1

---

## Step 7 — Evaluation

Create `eval/benchmark_vlm.json` — 20 images with known findings:

```json
{
  "image_id": "MURA_XR_WRIST_patient00042_study1_image1",
  "expected_findings": ["cortical thinning", "reduced trabecular density"],
  "expected_retrieval_keywords": ["cortical porosity", "trabecular"],
  "notes": "Osteoporotic wrist — should trigger bone loss literature"
}
```

Create `eval/run_eval_vlm.py` — measures:
- **Report coverage**: fraction of expected findings present in VLM report
- **Retrieval Recall@k**: does cross-modal retrieval find relevant text chunks?
- **End-to-end**: does the final LLM answer address the image findings?

---

## Execution order

| Week | Work |
|---|---|
| 1 | Download MURA + RSNA Bone Age · build `images.db` schema · write ingestion pipeline |
| 2 | Set up LLaVA 1.6 via Ollama · write `describe_images.py` · run on 500 test images · review report quality |
| 3 | Integrate cross-modal retrieval · update `retriever.py` · end-to-end test |
| 4 | Gradio "Analyse Image" tab · VLM eval benchmark · fix report quality issues |
| 5 | LoRA fine-tuning of LLaVA on bone images — novel contribution for the paper |

---

## Novel contributions (for the paper)

1. **Unified embedding space** — SPECTER2 embeds VLM text reports so visual findings directly retrieve scientific literature, without a separate cross-modal bridge model.
2. **Bone-specific VLM fine-tuning** — LoRA-tuned LLaVA on a curated bone image dataset produces more precise radiological descriptions than a general-purpose VLM.
3. **Grounded visual reasoning** — the system does not just describe an image; it connects visual findings to the mechanical and clinical literature automatically via the shared embedding space.
