# Textbooks Ingestion Pipeline

This document explains how the textbook ingestion pipeline works: what sources are used, how files are organised, and how to operate it.

---

## Status

> **In progress** — pipeline under construction.

| Metric | Value |
|---|---|
| **Textbooks collected** | 1 |
| **Source** | NCBI Bookshelf |

---

## Data Flow

```
Source (NCBI Bookshelf / manual)  →  raw PDF  →  data/raw/textbooks/  →  SQLite metadata
```

---

## Storage Layout

```
data/
└── raw/
    └── textbooks/
        └── <source>_<id>.pdf    e.g. ncbi_NBK45513.pdf
```

---

## Sources

| Source | Access | Notes |
|---|---|---|
| NCBI Bookshelf | Free, no login | Limited bone-specific titles; API-accessible |
| Manual | User-supplied | Any PDF dropped into `data/raw/textbooks/` |

---

## Running the Pipeline

> Commands will be added here once the pipeline is built.
