"""
reasoning/graph_db.py
=====================
SQLite-backed persistence for the bone knowledge graph.

Why SQLite?
-----------
The BoneKnowledgeGraph lives in memory (NetworkX).  OntologyStore is the
bridge that loads it from disk on startup and saves changes back.  SQLite
is the right choice for the same reasons as the rest of the project: zero
configuration, single-file, and Python-native.

Schema
------
nodes   — one row per concept; node_id is the primary key.
edges   — one row per directed relationship; composite key
          (source_node, relation, target_node).

Usage::

    from reasoning.graph_db import OntologyStore

    with OntologyStore() as store:
        store.upsert_node(node)
        store.upsert_edge(edge)
        graph = store.load_graph()
        print(store.stats())
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from config.settings import ONTOLOGY_DB_PATH
from reasoning.ontology import BoneKnowledgeGraph, Edge, Node

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS extraction_progress (
    chunk_id     TEXT PRIMARY KEY,
    status       TEXT NOT NULL,      -- 'done' | 'empty' | 'failed'
    n_triples    INTEGER DEFAULT 0,  -- triples successfully written for this chunk
    processed_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id     TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    node_type   TEXT NOT NULL,           -- structure, property, process, …
    description TEXT    DEFAULT '',
    source      TEXT    DEFAULT 'seed',  -- seed | extracted | manual
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id     TEXT PRIMARY KEY,        -- '<source>|<relation>|<target>'
    source_node TEXT NOT NULL REFERENCES nodes(node_id),
    target_node TEXT NOT NULL REFERENCES nodes(node_id),
    relation    TEXT NOT NULL,           -- determines | increases | …
    weight      REAL    DEFAULT 1.0,     -- confidence / co-occurrence frequency
    evidence    TEXT    DEFAULT '[]',    -- JSON list of chunk_ids from chunks.db
    edge_source TEXT    DEFAULT 'seed',  -- seed | extracted | manual
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node);
CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(node_type);
"""


class OntologyStore:
    """
    Context-manager interface to the SQLite ontology database.

    Typical usage::

        with OntologyStore() as store:
            store.upsert_node(node)
            store.upsert_edge(edge)
            graph = store.load_graph()
            print(store.stats())

    The context manager guarantees the connection is closed cleanly even if
    an exception occurs inside the ``with`` block.
    """

    def __init__(self, db_path: Path = ONTOLOGY_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the SQLite connection and create tables if they do not exist."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_TABLES_SQL)
        self._conn.commit()
        logger.debug("Connected to ontology store at %s", self.db_path)

    def close(self) -> None:
        """Flush pending writes and close the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> OntologyStore:
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the live connection, raising if connect() was not called."""
        if self._conn is None:
            raise RuntimeError("Call connect() or use as context manager first.")
        return self._conn

    # ── Write — nodes ─────────────────────────────────────────────────────────

    def upsert_node(self, node: Node) -> bool:
        """
        Insert a node, or update its label / description / type if it already
        exists (node_id collision).

        Parameters
        ----------
        node : Node
            The concept to persist.

        Returns
        -------
        bool
            True if this was a new insert; False if an existing row was updated.
        """
        existing = self.get_node(node.node_id)
        self.conn.execute(
            """
            INSERT INTO nodes (node_id, label, node_type, description, source)
            VALUES (:node_id, :label, :node_type, :description, :source)
            ON CONFLICT(node_id) DO UPDATE SET
                label       = excluded.label,
                node_type   = excluded.node_type,
                description = excluded.description
            """,
            {
                "node_id":     node.node_id,
                "label":       node.label,
                "node_type":   node.node_type,
                "description": node.description,
                "source":      node.source,
            },
        )
        self.conn.commit()
        return existing is None

    def upsert_nodes(self, nodes: list[Node]) -> tuple[int, int]:
        """
        Upsert a batch of nodes.

        Returns
        -------
        tuple[int, int]
            (new_count, updated_count).
        """
        new, updated = 0, 0
        for node in nodes:
            if self.upsert_node(node):
                new += 1
            else:
                updated += 1
        logger.info("upsert_nodes: %d new, %d updated", new, updated)
        return new, updated

    # ── Write — edges ─────────────────────────────────────────────────────────

    def upsert_edge(self, edge: Edge) -> bool:
        """
        Insert an edge, or update its weight and evidence list if the same
        (source, relation, target) triple already exists.

        When the same relationship is extracted from multiple chunks its
        weight accumulates — more evidence → higher confidence.

        Parameters
        ----------
        edge : Edge
            The relationship to persist.

        Returns
        -------
        bool
            True if this was a new insert; False if weight was updated.
        """
        existing = self.get_edge(edge.source, edge.relation, edge.target)

        if existing:
            # Merge evidence lists and increment weight.
            old_evidence: list[str] = json.loads(existing["evidence"] or "[]")
            merged = list(dict.fromkeys(old_evidence + edge.evidence))  # dedup, preserve order
            new_weight = existing["weight"] + edge.weight
            self.conn.execute(
                """
                UPDATE edges
                SET weight = :weight, evidence = :evidence
                WHERE edge_id = :edge_id
                """,
                {
                    "weight":   new_weight,
                    "evidence": json.dumps(merged),
                    "edge_id":  edge.edge_id,
                },
            )
        else:
            self.conn.execute(
                """
                INSERT INTO edges
                    (edge_id, source_node, target_node, relation,
                     weight, evidence, edge_source)
                VALUES
                    (:edge_id, :source, :target, :relation,
                     :weight, :evidence, :edge_source)
                """,
                {
                    "edge_id":     edge.edge_id,
                    "source":      edge.source,
                    "target":      edge.target,
                    "relation":    edge.relation,
                    "weight":      edge.weight,
                    "evidence":    json.dumps(edge.evidence),
                    "edge_source": edge.edge_source,
                },
            )
        self.conn.commit()
        return existing is None

    def upsert_edges(self, edges: list[Edge]) -> tuple[int, int]:
        """
        Upsert a batch of edges.

        Returns
        -------
        tuple[int, int]
            (new_count, updated_count).
        """
        new, updated = 0, 0
        for edge in edges:
            if self.upsert_edge(edge):
                new += 1
            else:
                updated += 1
        logger.info("upsert_edges: %d new, %d updated", new, updated)
        return new, updated

    # ── Read — nodes ──────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> sqlite3.Row | None:
        """Return the database row for node_id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        )
        return cur.fetchone()

    def get_all_nodes(self) -> list[sqlite3.Row]:
        """Return all node rows."""
        cur = self.conn.execute("SELECT * FROM nodes ORDER BY node_type, node_id")
        return cur.fetchall()

    # ── Read — edges ──────────────────────────────────────────────────────────

    def get_edge(
        self, source: str, relation: str, target: str
    ) -> sqlite3.Row | None:
        """Return the edge row for (source, relation, target), or None."""
        cur = self.conn.execute(
            "SELECT * FROM edges WHERE edge_id = ?",
            (f"{source}|{relation}|{target}",),
        )
        return cur.fetchone()

    def get_all_edges(self) -> list[sqlite3.Row]:
        """Return all edge rows."""
        cur = self.conn.execute("SELECT * FROM edges ORDER BY source_node, relation")
        return cur.fetchall()

    # ── Graph reconstruction ──────────────────────────────────────────────────

    def load_graph(self) -> BoneKnowledgeGraph:
        """
        Reconstruct a BoneKnowledgeGraph from the database.

        Nodes are loaded first, then edges.  Edges whose source or target
        is not present in the nodes table are skipped with a warning (this
        should not happen with a well-maintained database but guards against
        partial imports).

        Returns
        -------
        BoneKnowledgeGraph
            Fully populated in-memory graph.
        """
        graph = BoneKnowledgeGraph()

        node_rows = self.get_all_nodes()
        for row in node_rows:
            graph.add_node(
                Node(
                    node_id=row["node_id"],
                    label=row["label"],
                    node_type=row["node_type"],
                    description=row["description"] or "",
                    source=row["source"] or "seed",
                )
            )
        logger.info("Loaded %d nodes from ontology.db", len(node_rows))

        edge_rows = self.get_all_edges()
        skipped = 0
        for row in edge_rows:
            src, tgt = row["source_node"], row["target_node"]
            if src not in graph or tgt not in graph:
                logger.warning(
                    "Skipping edge %s — endpoint not in graph", row["edge_id"]
                )
                skipped += 1
                continue
            graph.add_edge(
                Edge(
                    source=src,
                    relation=row["relation"],
                    target=tgt,
                    weight=row["weight"],
                    evidence=json.loads(row["evidence"] or "[]"),
                    edge_source=row["edge_source"] or "seed",
                )
            )
        logger.info(
            "Loaded %d edges from ontology.db (%d skipped)", len(edge_rows) - skipped, skipped
        )
        return graph

    def save_graph(self, graph: BoneKnowledgeGraph) -> None:
        """
        Persist all nodes and edges from a BoneKnowledgeGraph to the database.

        Uses upsert semantics — safe to call repeatedly without creating
        duplicates.

        Parameters
        ----------
        graph : BoneKnowledgeGraph
            The graph to persist.
        """
        self.upsert_nodes(list(graph.iter_nodes()))
        self.upsert_edges(list(graph.iter_edges()))
        logger.info("Saved graph to ontology.db: %s", graph)

    # ── Extraction progress ───────────────────────────────────────────────────

    def record_progress(
        self, chunk_id: str, status: str, n_triples: int = 0
    ) -> None:
        """
        Record that a chunk has been processed by the extractor.

        Parameters
        ----------
        chunk_id : str
            The chunks.db id of the processed chunk.
        status : str
            'done' — triples were extracted and written.
            'empty' — LLM returned no valid triples.
            'failed' — Ollama call failed or JSON could not be parsed.
        n_triples : int
            Number of triples successfully written (0 for empty/failed).
        """
        self.conn.execute(
            """
            INSERT INTO extraction_progress (chunk_id, status, n_triples)
            VALUES (:chunk_id, :status, :n_triples)
            ON CONFLICT(chunk_id) DO UPDATE SET
                status    = excluded.status,
                n_triples = excluded.n_triples,
                processed_at = datetime('now')
            """,
            {"chunk_id": chunk_id, "status": status, "n_triples": n_triples},
        )
        self.conn.commit()

    def get_processed_chunk_ids(self) -> set[str]:
        """Return the set of chunk_ids already processed (any status)."""
        cur = self.conn.execute("SELECT chunk_id FROM extraction_progress")
        return {row[0] for row in cur.fetchall()}

    def extraction_stats(self) -> dict[str, int]:
        """
        Return a breakdown of extraction progress.

        Returns
        -------
        dict with keys: done, empty, failed, total_processed, total_triples
        """
        row = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'done'   THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'empty'  THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END),
                COUNT(*),
                SUM(n_triples)
            FROM extraction_progress
            """
        ).fetchone()
        return {
            "done":            row[0] or 0,
            "empty":           row[1] or 0,
            "failed":          row[2] or 0,
            "total_processed": row[3] or 0,
            "total_triples":   row[4] or 0,
        }

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """
        Return a summary of database contents.

        Returns
        -------
        dict with keys:
            total_nodes    — number of nodes
            total_edges    — number of edges
            seed_nodes     — nodes from hand-curated seed data
            extracted_nodes — nodes from LLM extraction pipeline
            seed_edges     — edges from hand-curated seed data
            extracted_edges — edges from LLM extraction pipeline
        """
        rows = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM nodes)                              AS total_nodes,
                (SELECT COUNT(*) FROM edges)                              AS total_edges,
                (SELECT COUNT(*) FROM nodes WHERE source = 'seed')        AS seed_nodes,
                (SELECT COUNT(*) FROM nodes WHERE source = 'extracted')   AS extracted_nodes,
                (SELECT COUNT(*) FROM edges WHERE edge_source = 'seed')   AS seed_edges,
                (SELECT COUNT(*) FROM edges WHERE edge_source = 'extracted') AS extracted_edges
            """
        ).fetchone()
        return dict(rows)
