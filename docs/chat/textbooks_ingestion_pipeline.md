# Textbooks Ingestion Pipeline

This document explains how the textbook ingestion pipeline works: what sources are used, which books are included, how files are organised, and how to operate it.

---

## Status

**Phase 1b — Complete (April 2026)**

| Metric | Value |
|---|---|
| **Textbooks collected** | 16 |
| **Sources** | 5 (MDPI Books, OpenStax, NCBI Bookshelf, Open Textbook Library, Springer Open) |

---

## Data Flow

```
data/raw/textbooks/<Source>/<book>.pdf
            │
            ▼
  ingestion/textbooks/scanner.py
  (reads page count via PyMuPDF, derives title and source from folder/filename)
            │
            ▼
  data/db/textbooks.db  (SQLite — title, source, pages, size, status)
```

---

## Storage Layout

```
data/
└── raw/
    └── textbooks/
        ├── MDPI Books/
        ├── NCBI Bookshelf/
        ├── Open Textbook Library/
        ├── OpenStax/
        └── Springer Open/
```

Folder names are used as the **source** label in the database.

---

## Textbook Catalog

### MDPI Books (8 books)

| Title | Topic |
|---|---|
| Advances in Bone Graft Materials | Biomaterials, grafting |
| Advances in Soft Tissue and Bone Sarcoma | Pathology, oncology |
| Biomaterials for Bone Tissue Engineering | Biomaterials |
| Bone Regeneration and Repair Materials | Regeneration, repair |
| Current Status and Future Directions of Bone Trauma Surgery | Trauma, surgery |
| Design of Materials for Bone Tissue Scaffolds | Scaffolds, biomaterials |
| Feature Papers in Bone Biomaterials | Biomaterials review |
| Scaffolds and Implants for Bone Regeneration | Scaffolds, implants |

### NCBI Bookshelf (1 book)

| Title | Topic |
|---|---|
| Bone Health and Osteoporosis | Physiology, pathology |

### Open Textbook Library (4 books)

| Title | Topic |
|---|---|
| Comparative Vertebrate and Human Anatomy | Comparative anatomy |
| Fundamentals of Anatomy and Physiology | General A&P |
| Fundamentals of Human Anatomy Laboratory Manual | Anatomy lab reference |
| Introduction to Human Osteology | Osteology, skeletal anatomy |

### OpenStax (1 book)

| Title | Topic |
|---|---|
| Anatomy and Physiology 2e | Comprehensive A&P reference |

### Springer Open (2 books)

| Title | Topic |
|---|---|
| Musculoskeletal Diseases 2021–2024 | Clinical, musculoskeletal |
| Musculoskeletal Diseases 2026–2029 | Clinical, musculoskeletal |

---

## Source Selection Rationale

Books were manually curated from four open-access sources:

- **MDPI Books** — peer-reviewed edited volumes freely available under CC licence; strong coverage of bone biomaterials and regeneration
- **NCBI Bookshelf** — free biomedical reference texts; covers bone physiology and pathology
- **Open Textbook Library** — peer-reviewed open textbooks; covers anatomy, physiology, osteology
- **OpenStax** — high-quality, widely adopted open textbooks; *Anatomy and Physiology 2e* is the standard open-access reference
- **Springer Open** — fully open-access Springer titles; *Musculoskeletal Diseases* series covers clinical context

Books were curated manually rather than by automated search — textbook quality matters more than volume, and the total number of relevant open-access titles is small enough to curate by hand.

Redundant books removed during curation:
- Two duplicate sarcoma volumes (kept the most comprehensive)
- Two duplicate anatomy lab manuals (kept Fundamentals)
- A preparatory A&P course (superseded by OpenStax 2e)
- A general human biology textbook (too broad)

---

## Code

| File | Purpose |
|---|---|
| `ingestion/textbooks/storage.py` | SQLite schema and `TextbookStore` class |
| `ingestion/textbooks/scanner.py` | Scans folder, extracts metadata via PyMuPDF |
| `ingestion/textbooks/pipeline.py` | CLI entrypoint — runs the scan and prints summary |

---

## Running the Pipeline

```bash
# Install PyMuPDF if not already installed
pip install -r requirements.txt

# Scan all textbooks and register in database
python -m ingestion.textbooks.pipeline
```

The pipeline is safe to re-run — existing records are updated in place, no duplicates are created.

---

## Next Steps

- [ ] Phase 2: Extract full text from textbook PDFs (PyMuPDF)
- [ ] Chunk text into overlapping segments
- [ ] Embed chunks into the same vector store as papers (unified retrieval)
