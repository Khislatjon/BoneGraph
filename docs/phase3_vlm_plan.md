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

| Dataset | Modality | Content | Size | Access |
|---|---|---|---|---|
| **MURA** (Stanford) | X-ray | 7 upper extremity parts · normal/abnormal labels | 40K images | Free — register at stanfordmlgroup.github.io |
| **OAI** (NIH) | X-ray + MRI | Knee osteoarthritis · longitudinal cohort | 4,796 subjects | Free — register at nda.nih.gov |
| **RSNA Bone Age** | Hand X-ray | Paediatric bone age · Greulich-Pyle graded | 12K images | Kaggle, free |
| **VerSe** | CT | Vertebral labelling + segmentation | ~300 scans | GitHub, free |
| **Osteoporosis X-ray** | Spine/hip X-ray | Osteoporosis graded | ~5K | Kaggle, free |
| **TCIA — TCGA-SARC** | MRI / CT | Bone sarcoma | ~100 cases | Free — register at cancerimagingarchive.net |

**Start with:** MURA + RSNA Bone Age. Both are on Kaggle, no institutional approval needed, and cover the most common bone X-ray findings.

**Capture per image:**
- File path, modality (xray / mri / ct), body part (femur, spine, hand, knee, wrist…)
- Pathology labels if provided (normal, abnormal, osteoporosis, fracture)
- Source dataset name, subject ID

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
