# Bone knowledge graph — interactive visualisations

Three HTML views of the cleaned + reclassified bone knowledge graph in
`data/db/ontology.db`. All three are pyvis force-directed layouts: drag to
pan, scroll to zoom, hover a node for label / type / degree / description.

| File | Nodes | Edges | What it shows |
|---|---:|---:|---|
| [`ontology_full.html`](ontology_full.html) | 1597 | 1584 | Everything. Dense — useful for filtering by type in the in-page control panel. |
| [`ontology_hubs.html`](ontology_hubs.html) | 577 | 906 | Top-80 nodes by degree + their direct neighbours. The "load-bearing" subgraph. |
| [`ontology_seed.html`](ontology_seed.html) | 140 | 255 | Only the hand-curated seed backbone. Most readable. |

## Node-type colour legend

| Colour | Type |
|---|---|
| ![#4C78A8](https://placehold.co/12x12/4C78A8/4C78A8.png) `#4C78A8` | structure |
| ![#F58518](https://placehold.co/12x12/F58518/F58518.png) `#F58518` | property |
| ![#54A24B](https://placehold.co/12x12/54A24B/54A24B.png) `#54A24B` | process |
| ![#E45756](https://placehold.co/12x12/E45756/E45756.png) `#E45756` | pathology |
| ![#72B7B2](https://placehold.co/12x12/72B7B2/72B7B2.png) `#72B7B2` | mechanism |
| ![#B279A2](https://placehold.co/12x12/B279A2/B279A2.png) `#B279A2` | material |
| ![#EECA3B](https://placehold.co/12x12/EECA3B/EECA3B.png) `#EECA3B` | factor |
| ![#FF9DA6](https://placehold.co/12x12/FF9DA6/FF9DA6.png) `#FF9DA6` | clinical |
| ![#9D755D](https://placehold.co/12x12/9D755D/9D755D.png) `#9D755D` | scale |
| ![#BAB0AC](https://placehold.co/12x12/BAB0AC/BAB0AC.png) `#BAB0AC` | cell |
| ![#D3D3D3](https://placehold.co/12x12/D3D3D3/D3D3D3.png) `#D3D3D3` | concept (catch-all) |

Node size scales with degree, so hubs stand out. Edge colour encodes the
relation (`increases`, `decreases`, `activates`, `inhibits`, `determines`,
`leads_to`, `is_part_of`, `predicts`, `analogous_to`, `correlates_with`,
`measures`); hover an edge for the relation name and its weight.

## Regenerating

```bash
python -m reasoning.visualize_ontology
```

Reads `data/db/ontology.db` and overwrites all three HTML files in this
directory. See [`reasoning/visualize_ontology.py`](../../reasoning/visualize_ontology.py)
for the layout / palette settings.
