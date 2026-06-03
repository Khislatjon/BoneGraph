"""
reasoning/visualize_ontology.py
================================
Render the bone knowledge graph to interactive HTML using pyvis.

Outputs three views to data/graph_viz/:
    ontology_full.html    all nodes/edges (filter in-page)
    ontology_seed.html    only hand-curated seed nodes/edges
    ontology_hubs.html    top-N nodes by total degree

Run::
    python -m reasoning.visualize_ontology
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from reasoning.graph_db import OntologyStore
from reasoning.ontology import BoneKnowledgeGraph

logger = logging.getLogger(__name__)

OUT_DIR = Path("visualisation/graph")

# Distinct, colour-blind-friendly palette for the 11 node types.
TYPE_COLOR: dict[str, str] = {
    "structure":  "#4C78A8",
    "property":   "#F58518",
    "process":    "#54A24B",
    "pathology":  "#E45756",
    "mechanism":  "#72B7B2",
    "material":   "#B279A2",
    "factor":     "#EECA3B",
    "clinical":   "#FF9DA6",
    "scale":      "#9D755D",
    "cell":       "#BAB0AC",
    "concept":    "#D3D3D3",
}

RELATION_STYLE: dict[str, str] = {
    "increases":       "#2ca02c",
    "decreases":       "#d62728",
    "activates":       "#17becf",
    "inhibits":        "#e377c2",
    "determines":      "#1f77b4",
    "leads_to":        "#ff7f0e",
    "is_part_of":      "#7f7f7f",
    "predicts":        "#9467bd",
    "analogous_to":    "#8c564b",
    "correlates_with": "#bcbd22",
    "measures":        "#000000",
}


def _to_networkx(graph: BoneKnowledgeGraph) -> nx.DiGraph:
    """Return the underlying nx.DiGraph (with attrs we set on add)."""
    return graph._graph  # noqa: SLF001 — intentional; viz is a friend module


def _render(nx_graph: nx.DiGraph, out_file: Path, title: str) -> None:
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0e1117",
        font_color="#e6e6e6",
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )
    # Force-directed layout that handles a few hundred nodes well.
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.9,
    )

    degrees = dict(nx_graph.degree())

    for node_id, data in nx_graph.nodes(data=True):
        ntype = data.get("node_type", "concept")
        label = data.get("label", node_id)
        desc  = data.get("description", "") or ""
        deg   = degrees.get(node_id, 0)
        size  = 8 + 2.5 * (deg ** 0.6)  # mild scaling so hubs stand out

        title_html = (
            f"<b>{label}</b><br>"
            f"<i>{ntype}</i> &middot; degree {deg}<br>"
            f"<span style='color:#aaa'>{desc[:240]}</span>"
        )
        net.add_node(
            node_id,
            label=label,
            title=title_html,
            color=TYPE_COLOR.get(ntype, "#888"),
            size=size,
            borderWidth=1,
        )

    for u, v, data in nx_graph.edges(data=True):
        rel = data.get("relation", "")
        w   = data.get("weight", 1.0)
        net.add_edge(
            u, v,
            title=f"{rel} (w={w:.2f})",
            color=RELATION_STYLE.get(rel, "#888"),
            width=min(1.0 + 0.5 * w, 6.0),
            arrows="to",
        )

    # Built-in control panel so you can tweak physics / filter live.
    net.show_buttons(filter_=["physics", "nodes", "edges"])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_file), notebook=False, open_browser=False)
    logger.info("Wrote %s (%d nodes, %d edges) — %s",
                out_file, nx_graph.number_of_nodes(),
                nx_graph.number_of_edges(), title)


def render_views(top_n_hubs: int = 80) -> None:
    with OntologyStore() as store:
        graph = store.load_graph()

    g = _to_networkx(graph)

    # 1) Full graph
    _render(g, OUT_DIR / "ontology_full.html", "Full ontology")

    # 2) Seed-only backbone (hand-curated)
    seed_nodes = [
        n for n, d in g.nodes(data=True) if d.get("source") == "seed"
    ]
    seed_g = g.subgraph(seed_nodes).copy()
    _render(seed_g, OUT_DIR / "ontology_seed.html", "Seed backbone")

    # 3) Top-N hubs + their direct neighbours
    degree = dict(g.degree())
    hubs = sorted(degree, key=degree.get, reverse=True)[:top_n_hubs]
    keep = set(hubs)
    for h in hubs:
        keep.update(g.predecessors(h))
        keep.update(g.successors(h))
    hubs_g = g.subgraph(keep).copy()
    _render(hubs_g, OUT_DIR / "ontology_hubs.html",
            f"Top-{top_n_hubs} hubs + neighbours")

    # Legend printed to stdout for convenience
    print("\nNode-type colours:")
    for t, c in TYPE_COLOR.items():
        print(f"  {c}  {t}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_views()
