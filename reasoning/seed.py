"""
reasoning/seed.py
=================
Hand-curated seed data for the BoneLogic bone knowledge graph.

Content
-------
SEED_NODES  168 core bone science concepts across 10 typed categories.
SEED_EDGES  ~220 hand-curated causal and structural relationships.

These represent the highest-confidence, domain-expert-validated starting
point for the graph.  The automated triple extraction pipeline (Step 4.2)
will add thousands more nodes and edges from the 248,629-chunk corpus;
the seed layer ensures the core causal backbone is always present and
correct regardless of extraction quality.

Running this module directly bootstraps ontology.db from scratch::

    python -m reasoning.seed           # populate (idempotent — safe to re-run)
    python -m reasoning.seed --stats   # print graph stats after seeding

Design decisions
----------------
* node_id uses snake_case throughout for consistency with automated
  extraction outputs.
* Descriptions are kept to one or two sentences — enough to ground LLM
  reasoning without being verbose.
* Edge weights are all 1.0 for seed data.  Extracted edges accumulate
  weight proportionally to the number of corpus chunks that support them,
  so seed edges will eventually be dominated by extraction evidence on
  well-studied relationships.
* Edges only reference nodes that are defined in SEED_NODES — the
  bootstrap script validates this before writing to the database.
"""

from __future__ import annotations

import argparse
import logging
import sys

from reasoning.graph_db import OntologyStore
from reasoning.ontology import BoneKnowledgeGraph, Edge, Node

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Seed nodes ────────────────────────────────────────────────────────────────
# Each entry: (node_id, label, node_type, description)

_RAW_NODES: list[tuple[str, str, str, str]] = [
    # ── Structures ────────────────────────────────────────────────────────────
    ("cortical_bone",            "cortical bone",             "structure", "Dense outer layer of bone forming the diaphysis and cortex of long bones; provides primary mechanical strength."),
    ("trabecular_bone",          "trabecular bone",           "structure", "Porous inner network of struts and plates (spongy bone); dominant in vertebral bodies and metaphyses."),
    ("osteon",                   "osteon",                    "structure", "Basic structural unit of cortical bone (Haversian system); cylindrical lamellar structure surrounding a central canal."),
    ("Haversian_canal",          "Haversian canal",           "structure", "Central canal within an osteon containing blood vessels, nerves, and lymphatics."),
    ("osteocyte_lacuna",         "osteocyte lacuna",          "structure", "Ellipsoidal cavity within mineralised bone matrix housing a single osteocyte."),
    ("canalicular_network",      "canalicular network",       "structure", "Interconnected microscale channels linking osteocyte lacunae and enabling cell-to-cell communication and fluid transport."),
    ("collagen_fibril",          "collagen fibril",           "structure", "Nanoscale rope-like type I collagen fibre providing tensile toughness to bone."),
    ("mineralized_collagen_fibril", "mineralised collagen fibril", "structure", "Collagen fibril with hydroxyapatite platelets deposited within and around it; the basic nanoscale building block of bone."),
    ("hydroxyapatite_crystal",   "hydroxyapatite crystal",    "structure", "Calcium phosphate mineral (Ca₁₀(PO₄)₆(OH)₂) that confers stiffness and hardness to bone tissue."),
    ("mineral_platelet",         "mineral platelet",          "structure", "Flat nanoscale plate of carbonate-substituted hydroxyapatite intercalated between collagen molecules."),
    ("lamella",                  "lamella",                   "structure", "Thin (3–7 µm) concentric layer of mineralised collagen fibrils with alternating fibril orientation."),
    ("interstitial_lamella",     "interstitial lamella",      "structure", "Remnant lamellar bone between osteons, representing older unremodelled tissue with higher mineralisation."),
    ("circumferential_lamella",  "circumferential lamella",   "structure", "Lamellar layers encircling the entire outer (periosteal) or inner (endosteal) bone surface."),
    ("cement_line",              "cement line",               "structure", "Reversal or arrest line at the boundary of an osteon; mineral-rich, collagen-poor interface that deflects cracks."),
    ("microcrack",               "microcrack",                "structure", "Submillimetre crack in bone matrix arising from fatigue loading or acute overload; accumulates with age and disease."),
    ("periosteum",               "periosteum",                "structure", "Fibrous membrane covering the outer bone surface; contains osteoprogenitor cells and is involved in periosteal apposition."),
    ("endosteum",                "endosteum",                 "structure", "Thin cellular membrane lining marrow cavity, trabecular surfaces, and Haversian canals."),
    ("growth_plate",             "growth plate",              "structure", "Cartilaginous zone (physis) at the ends of long bones driving longitudinal growth via endochondral ossification."),
    ("bone_marrow",              "bone marrow",               "structure", "Soft tissue filling the medullary cavity; contains haematopoietic stem cells and adipocytes."),
    ("trabecular_strut",         "trabecular strut",          "structure", "Individual rod- or plate-like element within the trabecular network; thickness and connectivity determine strength."),
    ("collagen_crosslink",       "collagen crosslink",        "structure", "Biochemical bond between adjacent collagen molecules stabilising fibril structure; enzymatic and non-enzymatic forms exist."),
    ("osteoid",                  "osteoid",                   "structure", "Unmineralised collagen matrix secreted by osteoblasts prior to mineralisation; a transient pre-bone tissue."),
    ("woven_bone",               "woven bone",                "structure", "Rapidly deposited immature bone with randomly oriented collagen fibrils; forms in fracture callus and foetal skeleton."),
    ("lamellar_bone",            "lamellar bone",             "structure", "Mature bone with highly organised parallel collagen lamellae; mechanically superior to woven bone."),
    ("subchondral_bone",         "subchondral bone",          "structure", "Bone immediately beneath articular cartilage; transfers loads between cartilage and the underlying trabecular network."),
    ("collagen_matrix",          "collagen matrix",           "structure", "Three-dimensional network of type I collagen providing the organic scaffold of bone."),
    ("cortical_shell",           "cortical shell",            "structure", "Thin cortical envelope surrounding the cancellous core of vertebral bodies; critical for vertebral load transmission."),
    ("Volkmann_canal",           "Volkmann canal",            "structure", "Transverse perforating canal connecting Haversian canals and the periosteum/endosteum, carrying blood vessels."),
    ("secondary_osteon",         "secondary osteon",          "structure", "Osteon formed by intracortical remodelling, replacing pre-existing bone with a new lamellar cylinder."),
    ("bone_fluid",               "bone fluid",                "structure", "Interstitial fluid within the lacunar-canalicular network; flow under loading is sensed by osteocytes."),

    # ── Properties ────────────────────────────────────────────────────────────
    ("elastic_modulus",          "elastic modulus",           "property",  "Stiffness of bone tissue; ratio of stress to strain in the elastic region (GPa). Anisotropic: higher in longitudinal direction."),
    ("fracture_toughness",       "fracture toughness",        "property",  "Resistance to crack propagation (MPa√m); governed by both collagen integrity and toughening mechanisms such as crack deflection."),
    ("yield_strength",           "yield strength",            "property",  "Stress at onset of permanent (plastic) deformation in bone (MPa)."),
    ("ultimate_strength",        "ultimate strength",         "property",  "Maximum stress bone can sustain before failure (MPa); integrates elastic and plastic response."),
    ("compressive_strength",     "compressive strength",      "property",  "Maximum compressive stress before failure; higher in cortical than trabecular bone and direction-dependent."),
    ("tensile_strength",         "tensile strength",          "property",  "Maximum tensile stress before failure; lower than compressive strength due to crack sensitivity."),
    ("fatigue_life",             "fatigue life",              "property",  "Number of loading cycles to failure at a given stress amplitude; decreases with porosity and damage accumulation."),
    ("bone_mineral_density",     "bone mineral density",      "property",  "Areal (DXA) or volumetric (QCT) mineral content (g/cm²); primary clinical surrogate for bone strength."),
    ("porosity",                 "porosity",                  "property",  "Volume fraction of void space in bone tissue (%); includes Haversian canals, lacunae, and resorption spaces."),
    ("anisotropy",               "anisotropy",                "property",  "Directional dependence of mechanical properties; cortical bone is ~40% stiffer longitudinally than transversely."),
    ("viscoelasticity",          "viscoelasticity",           "property",  "Time-dependent mechanical behaviour of bone; stiffness and strength increase with strain rate."),
    ("hardness",                 "hardness",                  "property",  "Resistance to localised plastic deformation; measured by nanoindentation (GPa) and correlated with mineralisation."),
    ("stiffness",                "stiffness",                 "property",  "Structural resistance to deformation (force/displacement); depends on both material modulus and geometry."),
    ("toughness",                "toughness",                 "property",  "Total energy absorbed per unit volume before fracture (J/m²); requires both strength and ductility."),
    ("collagen_crosslink_density","collagen crosslink density","property", "Number of inter-molecular collagen bonds per unit volume; enzymatic crosslinks increase toughness; AGE crosslinks reduce it."),
    ("mineral_crystallinity",    "mineral crystallinity",     "property",  "Degree of crystalline order in hydroxyapatite assessed by FTIR or XRD; increases with secondary mineralisation and age."),
    ("degree_of_mineralization", "degree of mineralisation",  "property",  "Mass fraction of mineral in bone tissue; principal determinant of stiffness and hardness."),
    ("apparent_density",         "apparent density",          "property",  "Mass of bone tissue per bulk volume including pores (g/cm³); relates to compressive strength by a power law."),
    ("tissue_mineral_density",   "tissue mineral density",    "property",  "Mineral content per unit bone tissue volume excluding pores; higher resolution than apparent density."),
    ("cortical_thickness",       "cortical thickness",        "property",  "Thickness of the cortical shell (mm); major determinant of whole-bone bending and torsional strength."),
    ("trabecular_thickness",     "trabecular thickness",      "property",  "Average thickness of individual trabeculae (µm); thinner trabeculae buckle at lower loads."),
    ("trabecular_spacing",       "trabecular spacing",        "property",  "Mean centre-to-centre distance between trabeculae (µm); increases with bone loss."),
    ("trabecular_connectivity",  "trabecular connectivity",   "property",  "Degree of interconnection within the trabecular network; loss of connectivity disproportionately reduces strength."),
    ("collagen_orientation",     "collagen orientation",      "property",  "Predominant angle of collagen fibrils relative to the bone axis; determines anisotropy pattern."),
    ("lacunar_density",          "lacunar density",           "property",  "Number of osteocyte lacunae per unit bone area; lacunae act as stress concentrators under cyclic loading."),
    ("vascular_porosity",        "vascular porosity",         "property",  "Porosity fraction attributable to Haversian and Volkmann canals; increases with intracortical remodelling rate."),
    ("fatigue_crack_growth_rate","fatigue crack growth rate", "property",  "Rate of crack extension per loading cycle (m/cycle) described by Paris law: da/dN = C·ΔKᵐ."),
    ("creep_rate",               "creep rate",                "property",  "Rate of time-dependent strain under sustained constant load; relevant to long-duration weight-bearing."),

    # ── Processes ─────────────────────────────────────────────────────────────
    ("bone_remodelling",         "bone remodelling",          "process",   "Coordinated coupled cycle of osteoclastic resorption followed by osteoblastic formation; renews ~10% of skeleton per year."),
    ("mineralisation",           "mineralisation",            "process",   "Deposition of hydroxyapatite into osteoid collagen matrix; primary mineralisation is rapid, secondary mineralisation continues over months."),
    ("crack_propagation",        "crack propagation",         "process",   "Advancement of a crack through bone tissue under applied or residual stress; rate governed by Paris law."),
    ("fatigue_crack_initiation", "fatigue crack initiation",  "process",   "Nucleation of a crack under cyclic loading below the yield stress; occurs preferentially at lacunae and cement lines."),
    ("microdamage_accumulation", "microdamage accumulation",  "process",   "Progressive build-up of microcracks from repeated subthreshold loading; leads to stiffness loss and eventual fatigue fracture."),
    ("osteoclast_resorption",    "osteoclast resorption",     "process",   "Enzymatic dissolution of mineralised bone matrix by osteoclasts; creates resorption lacunae (Howship's lacunae)."),
    ("osteoblast_formation",     "osteoblast formation",      "process",   "Synthesis and secretion of osteoid (type I collagen matrix) by osteoblasts, followed by mineralisation."),
    ("bone_formation",           "bone formation",            "process",   "Net increase in bone tissue mass through osteoblastic activity; includes osteoid synthesis and mineralisation."),
    ("bone_resorption",          "bone resorption",           "process",   "Net loss of bone tissue mass through osteoclastic activity; increases porosity and reduces mineral density."),
    ("secondary_mineralisation", "secondary mineralisation",  "process",   "Slow continued increase in mineral content and crystallinity of existing bone matrix over months to years."),
    ("crack_deflection",         "crack deflection",          "process",   "Redirection of crack path around cement lines and other microstructural interfaces; dissipates fracture energy."),
    ("crack_bridging",           "crack bridging",            "process",   "Intact collagen fibrils or osteon fibres spanning a crack wake, exerting closure forces on the crack faces."),
    ("collagen_crosslinking",    "collagen crosslinking",     "process",   "Formation of enzymatic inter-molecular pyridinoline and deoxypyridinoline bonds stabilising collagen fibrils."),
    ("non_enzymatic_glycation",  "non-enzymatic glycation",   "process",   "Non-enzymatic attachment of reducing sugars (glucose) to collagen, forming advanced glycation end-products (AGEs); accumulates with age."),
    ("mechanotransduction",      "mechanotransduction",       "process",   "Conversion of mechanical signals (strain, fluid flow, pressure) into cellular biochemical responses governing adaptation."),
    ("fracture_healing",         "fracture healing",          "process",   "Sequential repair of bone through haematoma formation, soft callus, hard callus mineralisation, and remodelling."),
    ("ossification",             "ossification",              "process",   "Bone tissue formation; endochondral (via cartilage template) or intramembranous (direct from mesenchyme)."),
    ("apoptosis",                "apoptosis",                 "process",   "Programmed cell death; osteocyte apoptosis at microdamage sites signals targeted remodelling."),
    ("targeted_remodelling",     "targeted remodelling",      "process",   "Remodelling specifically directed at microdamage sites, triggered by osteocyte apoptosis and canalicular signalling."),
    ("stress_relaxation",        "stress relaxation",         "process",   "Decrease in stress under constant strain over time; a viscoelastic property of bone collagen."),
    ("creep",                    "creep",                     "process",   "Progressive deformation under sustained constant load; driven by collagen viscoelasticity."),
    ("piezoelectric_signaling",  "piezoelectric signalling",  "process",   "Generation of electrical signals in bone under mechanical deformation; proposed to contribute to mechanosensing."),

    # ── Pathologies ───────────────────────────────────────────────────────────
    ("osteoporosis",             "osteoporosis",              "pathology", "Systemic skeletal disease of low bone mass and microarchitectural deterioration; defined by T-score ≤ −2.5."),
    ("osteopenia",               "osteopenia",                "pathology", "Low bone mass (T-score between −1 and −2.5); an intermediate state with elevated but not osteoporotic fracture risk."),
    ("stress_fracture",          "stress fracture",           "pathology", "Fatigue fracture arising from repetitive subthreshold loading without adequate recovery time for remodelling."),
    ("fragility_fracture",       "fragility fracture",        "pathology", "Fracture from low-energy trauma (e.g. fall from standing height) due to compromised bone strength."),
    ("hip_fracture",             "hip fracture",              "pathology", "Fracture of the proximal femur; major osteoporotic fracture type with high morbidity and mortality in the elderly."),
    ("vertebral_fracture",       "vertebral fracture",        "pathology", "Compression fracture of a vertebral body; most common osteoporotic fracture, often clinically silent."),
    ("osteogenesis_imperfecta",  "osteogenesis imperfecta",   "pathology", "Genetic disorder of type I collagen causing brittle bone disease; ranges from mild to perinatally lethal."),
    ("Paget_disease",            "Paget disease",             "pathology", "Chronic metabolic bone disease with disorganised accelerated remodelling; produces enlarged, structurally inferior woven bone."),
    ("avascular_necrosis",       "avascular necrosis",        "pathology", "Bone death from interruption of blood supply; leads to trabecular collapse and joint destruction if untreated."),
    ("bone_metastasis",          "bone metastasis",           "pathology", "Secondary tumour in bone from a distant primary cancer; disrupts bone remodelling and causes pathological fracture."),
    ("osteoarthritis",           "osteoarthritis",            "pathology", "Degenerative joint disease with cartilage loss, subchondral bone sclerosis, and osteophyte formation."),
    ("rheumatoid_arthritis",     "rheumatoid arthritis",      "pathology", "Autoimmune inflammatory arthritis causing synovitis, periarticular bone erosion, and systemic bone loss."),
    ("osteomalacia",             "osteomalacia",              "pathology", "Defective mineralisation of osteoid in adult bone, typically due to vitamin D deficiency or phosphate wasting."),
    ("rickets",                  "rickets",                   "pathology", "Defective mineralisation of growth plate cartilage in children, causing bowing of long bones."),
    ("hyperparathyroidism",      "hyperparathyroidism",       "pathology", "Excess PTH leading to elevated bone resorption, hypercalcaemia, and reduced bone mineral density."),
    ("osteosarcoma",             "osteosarcoma",              "pathology", "Primary malignant bone tumour arising from osteoblast-lineage cells; commonest primary bone cancer in adolescents."),
    ("cortical_porosity_increase","cortical porosity increase","pathology","Age- or disease-related increase in intracortical pore volume; reduces elastic modulus and fracture toughness."),
    ("periprosthetic_fracture",  "periprosthetic fracture",   "pathology", "Fracture around an orthopaedic implant, often driven by stress shielding-induced bone loss."),

    # ── Mechanisms ────────────────────────────────────────────────────────────
    ("Wolff_law",                "Wolff's law",               "mechanism", "Bone adapts its mass and architecture to the prevailing mechanical environment; loading drives formation, unloading drives resorption."),
    ("Frost_mechanostat",        "Frost mechanostat",         "mechanism", "Strain-threshold model: bone remodels to maintain peak strain within a physiological 'mechanostat' window (~1000–3000 µε)."),
    ("crack_deflection_toughening","crack deflection toughening","mechanism","Extrinsic toughening mechanism: cement lines redirect crack paths, forcing longer crack trajectories and greater energy dissipation."),
    ("crack_bridging_toughening","crack bridging toughening", "mechanism", "Extrinsic toughening mechanism: intact collagen fibrils bridge crack faces in the crack wake, applying closing forces."),
    ("stress_shielding",         "stress shielding",          "mechanism", "Reduction of bone strain adjacent to a stiff implant; chronically reduced strain triggers resorption per the mechanostat."),
    ("minimum_effective_strain", "minimum effective strain",  "mechanism", "Strain threshold (~1000 µε) below which bone resorption is preferentially triggered in the Frost mechanostat model."),
    ("strain_energy_density_theory","strain energy density theory","mechanism","Theoretical framework proposing bone remodels to equalise strain energy density throughout its structure."),
    ("coupling_remodelling",     "coupling of remodelling",   "mechanism", "Biochemical linkage ensuring osteoclast resorption is followed by osteoblast formation at the same site."),
    ("intracortical_remodelling","intracortical remodelling", "mechanism", "Haversian remodelling process creating secondary osteons by tunnelling through cortical bone."),
    ("piezoelectric_effect",     "piezoelectric effect",      "mechanism", "Bone generates an electrical potential under mechanical stress due to the polar asymmetry of collagen."),
    ("fluid_flow_mechanosensing","fluid flow mechanosensing",  "mechanism", "Osteocytes sense canalicular fluid flow generated by bone deformation; primary mechanosensory mechanism."),
    ("endochondral_ossification","endochondral ossification",  "mechanism", "Bone formation via a cartilage intermediate; the process driving longitudinal growth of long bones."),
    ("contact_inhibition",       "contact inhibition",        "mechanism", "Inhibition of cell proliferation upon contact with adjacent cells; modulates bone cell population dynamics."),
    ("targeted_remodelling_mechanism","targeted remodelling mechanism","mechanism","Osteocyte apoptosis at damage sites signals BMU recruitment via RANKL upregulation and sclerostin downregulation."),

    # ── Clinical ──────────────────────────────────────────────────────────────
    ("fracture_risk",            "fracture risk",             "clinical",  "Probability of a fracture over a defined time period; the primary clinical endpoint for bone disease assessment."),
    ("fall_risk",                "fall risk",                 "clinical",  "Probability of a fall; the proximate mechanical cause of most fragility fractures in the elderly."),
    ("FRAX_score",               "FRAX score",                "clinical",  "WHO algorithm estimating 10-year probability of major osteoporotic fracture based on clinical risk factors and BMD."),
    ("DXA_measurement",          "DXA measurement",           "clinical",  "Dual-energy X-ray absorptiometry; the clinical standard for measuring areal bone mineral density (g/cm²)."),
    ("bone_strength",            "bone strength",             "clinical",  "Integrated mechanical competence of a whole bone under physiological loading; encompasses geometry and material quality."),
    ("fracture_load",            "fracture load",             "clinical",  "Force required to fracture a bone under defined loading conditions; measured ex vivo or estimated computationally."),
    ("safety_factor",            "safety factor",             "clinical",  "Ratio of fracture load to peak habitual load; values < 1 indicate imminent fracture risk."),
    ("energy_absorption",        "energy absorption",         "clinical",  "Total energy absorbed before fracture (area under load-displacement curve); governs impact resistance."),
    ("biomechanical_competence", "biomechanical competence",  "clinical",  "Ability of bone to resist fracture under the range of physiological and traumatic loads it encounters."),
    ("structural_rigidity",      "structural rigidity",       "clinical",  "Resistance to bending or torsional deformation of a whole bone; product of material stiffness and geometric moment of inertia."),
    ("load_bearing_capacity",    "load-bearing capacity",     "clinical",  "Maximum load a bone structure can sustain before failure."),
    ("impact_resistance",        "impact resistance",         "clinical",  "Ability to absorb energy from impact or dynamic loading; governed by toughness and energy absorption."),

    # ── Materials (biomaterials & bio-inspired) ───────────────────────────────
    ("hydroxyapatite_scaffold",  "hydroxyapatite scaffold",   "material",  "Synthetic hydroxyapatite ceramic scaffold for bone substitution; osteoconductive but brittle."),
    ("bioglass",                 "bioglass",                  "material",  "Bioactive silicate glass that forms a hydroxyapatite layer in vivo and bonds directly to bone."),
    ("PEEK_implant",             "PEEK implant",              "material",  "Polyetheretherketone implant; radiolucent and modulus-matched to cortical bone, reducing stress shielding."),
    ("collagen_membrane",        "collagen membrane",         "material",  "Resorbable collagen barrier membrane used in guided bone regeneration to exclude soft tissue."),
    ("calcium_phosphate_cement", "calcium phosphate cement",  "material",  "Injectable self-setting calcium phosphate paste for void filling; resorbable and osteoconductive."),
    ("bone_graft",               "bone graft",                "material",  "Autologous, allogeneic, or xenogeneic bone transplanted to fill defects; gold standard for bone repair."),
    ("titanium_implant",         "titanium implant",          "material",  "Ti or Ti-6Al-4V orthopaedic/dental implant; high strength and osseointegration, but high modulus causes stress shielding."),
    ("porous_scaffold",          "porous scaffold",           "material",  "Three-dimensional porous structure supporting bone ingrowth; pore size 100–500 µm optimal for vascularisation."),
    ("demineralized_bone_matrix","demineralized bone matrix",  "material",  "Acid-extracted bone retaining collagen and BMPs; osteoinductive and used as bone graft extender."),
    ("beta_tricalcium_phosphate","beta-tricalcium phosphate",  "material",  "Resorbable calcium phosphate ceramic (β-TCP) that is resorbed and replaced by new bone over months."),
    ("bioprinted_bone",          "bioprinted bone",           "material",  "Additively manufactured patient-specific bone scaffold with controlled porosity and geometry."),
    ("composite_scaffold",       "composite scaffold",        "material",  "Hybrid scaffold combining a polymer matrix (e.g. collagen, PLA) with a ceramic filler (e.g. HA, β-TCP)."),
    ("fibrin_scaffold",          "fibrin scaffold",           "material",  "Protein scaffold derived from fibrinogen and thrombin; promotes cell adhesion and angiogenesis."),
    ("chitosan_scaffold",        "chitosan scaffold",         "material",  "Polysaccharide-based scaffold with intrinsic antimicrobial properties and tunable degradation."),
    ("bone_morphogenetic_protein","bone morphogenetic protein","material",  "Osteoinductive growth factor (BMP-2, BMP-7) that drives mesenchymal stem cell differentiation to osteoblasts."),
    ("bioactive_coating",        "bioactive coating",         "material",  "Surface treatment (e.g. HA coating, plasma spray) promoting osseointegration of metallic implants."),
    ("ceramic_scaffold",         "ceramic scaffold",          "material",  "Sintered calcium phosphate or bioglass ceramic scaffold for bone replacement."),
    ("polylactic_acid_scaffold", "polylactic acid scaffold",  "material",  "Biodegradable PLA polymer scaffold that degrades over months as new bone is deposited."),
    ("zirconia_implant",         "zirconia implant",          "material",  "Yttria-stabilised zirconia ceramic implant; high fracture toughness and white aesthetic for dental applications."),
    ("fibre_reinforced_composite","fibre-reinforced composite","material",  "Engineering structural material with aligned fibres in a matrix — an analogue to cortical bone's osteon-collagen organisation."),

    # ── Scales ────────────────────────────────────────────────────────────────
    ("molecular_scale",          "molecular scale",           "scale",     "Length scale of 0.1–10 nm; collagen triple helix, hydroxyapatite unit cell, crosslinks."),
    ("nanoscale",                "nanoscale",                 "scale",     "Length scale of 10–500 nm; mineral platelets, collagen fibrils, mineralised collagen fibrils."),
    ("microscale",               "microscale",                "scale",     "Length scale of 1–500 µm; osteons, osteocyte lacunae, trabeculae, Haversian canals."),
    ("mesoscale",                "mesoscale",                 "scale",     "Length scale of 0.5–10 mm; cortical and trabecular architecture, cortical shell thickness."),
    ("macroscale",               "macroscale",                "scale",     "Length scale > 10 mm; whole bone geometry, external shape, cross-sectional dimensions."),

    # ── Cells ─────────────────────────────────────────────────────────────────
    ("osteoblast",               "osteoblast",                "cell",      "Bone-forming cell that synthesises type I collagen matrix and promotes mineralisation; differentiated from mesenchymal stem cells."),
    ("osteoclast",               "osteoclast",                "cell",      "Multinucleated bone-resorbing cell derived from haematopoietic precursors; dissolves mineral and degrades matrix."),
    ("osteocyte",                "osteocyte",                 "cell",      "Mature osteoblast embedded within osteocyte lacunae; primary mechanosensor of bone, orchestrates adaptive remodelling."),

    # ── Factors (signalling molecules, hormones, drugs, environmental inputs) ──
    ("RANKL",                    "RANKL",                     "factor",    "Receptor activator of NF-κB ligand; expressed by osteoblasts and osteocytes; essential for osteoclast differentiation."),
    ("OPG",                      "OPG",                       "factor",    "Osteoprotegerin; decoy receptor for RANKL produced by osteoblasts; inhibits osteoclastogenesis and bone resorption."),
    ("PTH",                      "PTH",                       "factor",    "Parathyroid hormone; regulates calcium homeostasis; intermittent PTH is anabolic, continuous PTH is catabolic to bone."),
    ("estrogen",                 "estrogen",                  "factor",    "Sex steroid hormone that inhibits osteoclast activity and promotes bone formation; deficiency drives postmenopausal bone loss."),
    ("sclerostin",               "sclerostin",                "factor",    "Wnt signalling inhibitor secreted by osteocytes under mechanical unloading; suppresses bone formation."),
    ("BMP2",                     "BMP-2",                     "factor",    "Bone morphogenetic protein 2; potent osteoinductive growth factor that drives osteoblast differentiation from mesenchymal precursors."),
    ("IGF1",                     "IGF-1",                     "factor",    "Insulin-like growth factor 1; stimulates osteoblast proliferation and collagen synthesis; mediates growth hormone effects on bone."),
    ("TGF_beta",                 "TGF-β",                     "factor",    "Transforming growth factor beta; released from matrix during resorption; couples resorption to formation by recruiting osteoblasts."),
    ("calcium_ion",              "calcium ion",               "factor",    "Principal ionic component of hydroxyapatite (Ca²⁺); plasma calcium levels regulated by PTH, vitamin D, and calcitonin."),
    ("vitamin_D",                "vitamin D",                 "factor",    "Steroid hormone (calcitriol) regulating intestinal calcium absorption and bone mineralisation; deficiency causes osteomalacia."),
    ("aging",                    "aging",                     "factor",    "Progressive biological and microstructural deterioration of bone with age; increases AGEs, porosity, and fracture risk."),
    ("mechanical_loading",       "mechanical loading",        "factor",    "External or muscular forces applied to the skeleton; the primary driver of adaptive bone remodelling."),
    ("glucocorticoid",           "glucocorticoid",            "factor",    "Corticosteroid drug (e.g. prednisolone) causing secondary osteoporosis at high or prolonged doses."),
    ("advanced_glycation_endproduct","advanced glycation end-product","factor","Non-enzymatic glycation product (AGE) accumulating in collagen with age and diabetes; stiffens but embrittles bone."),
    ("testosterone",             "testosterone",              "factor",    "Anabolic sex hormone in males; converted to estrogen in bone and stimulates periosteal bone formation."),
    ("bisphosphonate",           "bisphosphonate",            "factor",    "Anti-resorptive drug class that inhibits osteoclast activity; first-line treatment for osteoporosis."),
]

SEED_NODES: list[Node] = [
    Node(node_id=nid, label=label, node_type=ntype, description=desc, source="seed")
    for nid, label, ntype, desc in _RAW_NODES
]

# ── Seed edges ────────────────────────────────────────────────────────────────
# Each entry: (source, relation, target)
# All weights default to 1.0; evidence is empty (hand-curated).

_RAW_EDGES: list[tuple[str, str, str]] = [

    # ── Structural hierarchy ──────────────────────────────────────────────────
    ("mineral_platelet",              "is_part_of", "mineralized_collagen_fibril"),
    ("collagen_fibril",               "is_part_of", "mineralized_collagen_fibril"),
    ("mineralized_collagen_fibril",   "is_part_of", "lamella"),
    ("lamella",                       "is_part_of", "osteon"),
    ("interstitial_lamella",          "is_part_of", "cortical_bone"),
    ("circumferential_lamella",       "is_part_of", "cortical_bone"),
    ("osteon",                        "is_part_of", "cortical_bone"),
    ("Haversian_canal",               "is_part_of", "osteon"),
    ("cement_line",                   "is_part_of", "osteon"),
    ("Volkmann_canal",                "is_part_of", "cortical_bone"),
    ("secondary_osteon",              "is_part_of", "cortical_bone"),
    ("trabecular_strut",              "is_part_of", "trabecular_bone"),
    ("osteocyte_lacuna",              "is_part_of", "cortical_bone"),
    ("canalicular_network",           "is_part_of", "cortical_bone"),
    ("cortical_shell",                "is_part_of", "vertebral_fracture"),   # cortical shell → vertebral strength
    ("collagen_crosslink",            "is_part_of", "collagen_fibril"),
    ("hydroxyapatite_crystal",        "is_part_of", "mineralized_collagen_fibril"),
    ("collagen_matrix",               "is_part_of", "osteoid"),
    ("subchondral_bone",              "is_part_of", "osteoarthritis"),

    # ── Scale hierarchy ───────────────────────────────────────────────────────
    ("mineral_platelet",              "is_part_of", "nanoscale"),
    ("collagen_fibril",               "is_part_of", "nanoscale"),
    ("mineralized_collagen_fibril",   "is_part_of", "nanoscale"),
    ("osteocyte_lacuna",              "is_part_of", "microscale"),
    ("osteon",                        "is_part_of", "microscale"),
    ("trabecular_strut",              "is_part_of", "microscale"),
    ("cortical_bone",                 "is_part_of", "mesoscale"),
    ("trabecular_bone",               "is_part_of", "mesoscale"),
    ("cortical_shell",                "is_part_of", "mesoscale"),

    # ── Mineral & collagen → mechanical properties ────────────────────────────
    ("degree_of_mineralization",      "determines",  "elastic_modulus"),
    ("degree_of_mineralization",      "increases",   "stiffness"),
    ("degree_of_mineralization",      "increases",   "hardness"),
    ("degree_of_mineralization",      "increases",   "compressive_strength"),
    ("mineral_crystallinity",         "increases",   "hardness"),
    ("mineral_crystallinity",         "increases",   "stiffness"),
    ("mineral_crystallinity",         "decreases",   "fracture_toughness"),   # higher crystallinity = more brittle
    ("collagen_crosslink_density",    "determines",  "fracture_toughness"),
    ("collagen_crosslink_density",    "increases",   "yield_strength"),
    ("collagen_crosslink_density",    "increases",   "tensile_strength"),
    ("collagen_crosslink_density",    "increases",   "toughness"),
    ("collagen_orientation",          "determines",  "anisotropy"),
    ("anisotropy",                    "determines",  "elastic_modulus"),

    # ── Porosity & density → mechanical properties ────────────────────────────
    ("porosity",                      "decreases",   "elastic_modulus"),
    ("porosity",                      "decreases",   "yield_strength"),
    ("porosity",                      "decreases",   "compressive_strength"),
    ("porosity",                      "decreases",   "fracture_toughness"),
    ("porosity",                      "decreases",   "fatigue_life"),
    ("vascular_porosity",             "decreases",   "elastic_modulus"),
    ("vascular_porosity",             "decreases",   "compressive_strength"),
    ("apparent_density",              "determines",  "elastic_modulus"),
    ("apparent_density",              "determines",  "compressive_strength"),
    ("apparent_density",              "determines",  "yield_strength"),
    ("lacunar_density",               "increases",   "fatigue_crack_growth_rate"),   # lacunae = stress concentrators
    ("lacunar_density",               "decreases",   "elastic_modulus"),

    # ── Geometry → whole-bone strength ────────────────────────────────────────
    ("cortical_thickness",            "determines",  "bone_strength"),
    ("cortical_thickness",            "determines",  "fracture_load"),
    ("cortical_thickness",            "determines",  "structural_rigidity"),
    ("trabecular_connectivity",       "determines",  "compressive_strength"),
    ("trabecular_thickness",          "increases",   "compressive_strength"),
    ("trabecular_spacing",            "decreases",   "compressive_strength"),
    ("bone_mineral_density",          "determines",  "bone_strength"),
    ("bone_mineral_density",          "determines",  "fracture_load"),

    # ── Toughening mechanisms ─────────────────────────────────────────────────
    ("crack_deflection",              "increases",   "fracture_toughness"),
    ("crack_bridging",                "increases",   "fracture_toughness"),
    ("crack_deflection_toughening",   "increases",   "fracture_toughness"),
    ("crack_bridging_toughening",     "increases",   "fracture_toughness"),
    ("cement_line",                   "activates",   "crack_deflection"),
    ("cement_line",                   "activates",   "crack_deflection_toughening"),
    ("osteon",                        "activates",   "crack_bridging"),
    ("osteon",                        "activates",   "crack_bridging_toughening"),
    ("microcrack",                    "decreases",   "fracture_toughness"),
    ("microdamage_accumulation",      "decreases",   "fracture_toughness"),
    ("microdamage_accumulation",      "decreases",   "fatigue_life"),
    ("microdamage_accumulation",      "decreases",   "elastic_modulus"),

    # ── Fatigue / damage chain ────────────────────────────────────────────────
    ("fatigue_crack_initiation",      "leads_to",    "crack_propagation"),
    ("crack_propagation",             "leads_to",    "fragility_fracture"),
    ("microdamage_accumulation",      "activates",   "targeted_remodelling"),
    ("microdamage_accumulation",      "activates",   "apoptosis"),            # osteocyte apoptosis at damage
    ("apoptosis",                     "activates",   "targeted_remodelling"),
    ("targeted_remodelling",          "leads_to",    "fracture_healing"),

    # ── Mineralisation processes ──────────────────────────────────────────────
    ("mineralisation",                "increases",   "elastic_modulus"),
    ("mineralisation",                "increases",   "degree_of_mineralization"),
    ("mineralisation",                "increases",   "hardness"),
    ("secondary_mineralisation",      "increases",   "mineral_crystallinity"),
    ("secondary_mineralisation",      "increases",   "degree_of_mineralization"),
    ("secondary_mineralisation",      "increases",   "hardness"),
    ("secondary_mineralisation",      "decreases",   "fracture_toughness"),   # increased crystallinity = more brittle
    ("osteoid",                       "leads_to",    "mineralisation"),
    ("collagen_crosslinking",         "increases",   "collagen_crosslink_density"),
    ("collagen_crosslinking",         "increases",   "fracture_toughness"),
    ("collagen_crosslinking",         "activates",   "bone_formation"),

    # ── Non-enzymatic glycation (AGE) effects ─────────────────────────────────
    ("non_enzymatic_glycation",       "decreases",   "fracture_toughness"),
    ("non_enzymatic_glycation",       "decreases",   "collagen_crosslink_density"),
    ("non_enzymatic_glycation",       "increases",   "stiffness"),
    ("non_enzymatic_glycation",       "increases",   "hardness"),
    ("advanced_glycation_endproduct", "decreases",   "fracture_toughness"),
    ("advanced_glycation_endproduct", "increases",   "stiffness"),
    ("non_enzymatic_glycation",       "leads_to",    "advanced_glycation_endproduct"),

    # ── Bone remodelling cycle ────────────────────────────────────────────────
    ("mechanical_loading",            "activates",   "bone_remodelling"),
    ("mechanical_loading",            "activates",   "mechanotransduction"),
    ("mechanical_loading",            "inhibits",    "sclerostin"),
    ("Wolff_law",                     "determines",  "bone_remodelling"),
    ("Frost_mechanostat",             "determines",  "minimum_effective_strain"),
    ("minimum_effective_strain",      "activates",   "bone_remodelling"),
    ("bone_remodelling",              "leads_to",    "intracortical_remodelling"),
    ("intracortical_remodelling",     "leads_to",    "secondary_osteon"),
    ("intracortical_remodelling",     "increases",   "vascular_porosity"),
    ("coupling_remodelling",          "determines",  "bone_remodelling"),
    ("osteoclast_resorption",         "leads_to",    "bone_resorption"),
    ("osteoblast_formation",          "leads_to",    "bone_formation"),
    ("bone_resorption",               "decreases",   "bone_mineral_density"),
    ("bone_resorption",               "decreases",   "cortical_thickness"),
    ("bone_resorption",               "increases",   "porosity"),
    ("bone_resorption",               "decreases",   "trabecular_connectivity"),
    ("bone_formation",                "increases",   "bone_mineral_density"),
    ("bone_formation",                "increases",   "cortical_thickness"),
    ("bone_formation",                "decreases",   "porosity"),

    # ── Mechanotransduction signalling ────────────────────────────────────────
    ("osteocyte",                     "activates",   "mechanotransduction"),
    ("fluid_flow_mechanosensing",     "activates",   "mechanotransduction"),
    ("piezoelectric_effect",          "activates",   "mechanotransduction"),
    ("mechanotransduction",           "activates",   "bone_remodelling"),
    ("mechanotransduction",           "inhibits",    "sclerostin"),
    ("piezoelectric_signaling",       "activates",   "bone_formation"),
    ("bone_fluid",                    "activates",   "fluid_flow_mechanosensing"),

    # ── Biological signalling ─────────────────────────────────────────────────
    ("RANKL",                         "activates",   "osteoclast_resorption"),
    ("OPG",                           "inhibits",    "osteoclast_resorption"),
    ("OPG",                           "inhibits",    "RANKL"),
    ("PTH",                           "activates",   "bone_remodelling"),
    ("PTH",                           "activates",   "RANKL"),
    ("PTH",                           "activates",   "osteoclast_resorption"),
    ("estrogen",                      "inhibits",    "osteoclast_resorption"),
    ("estrogen",                      "inhibits",    "RANKL"),
    ("estrogen",                      "activates",   "bone_formation"),
    ("BMP2",                          "activates",   "bone_formation"),
    ("BMP2",                          "activates",   "ossification"),
    ("IGF1",                          "activates",   "bone_formation"),
    ("IGF1",                          "activates",   "collagen_crosslinking"),
    ("TGF_beta",                      "activates",   "bone_formation"),
    ("TGF_beta",                      "activates",   "coupling_remodelling"),
    ("sclerostin",                    "inhibits",    "bone_formation"),
    ("sclerostin",                    "inhibits",    "osteoblast_formation"),
    ("vitamin_D",                     "activates",   "mineralisation"),
    ("vitamin_D",                     "increases",   "calcium_ion"),
    ("calcium_ion",                   "determines",  "mineralisation"),
    ("testosterone",                  "activates",   "bone_formation"),
    ("testosterone",                  "inhibits",    "osteoclast_resorption"),
    ("osteoblast",                    "activates",   "bone_formation"),
    ("osteoblast",                    "activates",   "mineralisation"),
    ("osteoblast",                    "activates",   "OPG"),
    ("osteoblast",                    "activates",   "RANKL"),
    ("osteoclast",                    "activates",   "osteoclast_resorption"),
    ("osteocyte",                     "activates",   "targeted_remodelling"),
    ("osteocyte",                     "activates",   "sclerostin"),          # under unloading

    # ── Aging effects ─────────────────────────────────────────────────────────
    ("aging",                         "increases",   "non_enzymatic_glycation"),
    ("aging",                         "decreases",   "collagen_crosslink_density"),
    ("aging",                         "increases",   "porosity"),
    ("aging",                         "increases",   "vascular_porosity"),
    ("aging",                         "increases",   "cortical_porosity_increase"),
    ("aging",                         "decreases",   "bone_mineral_density"),
    ("aging",                         "decreases",   "cortical_thickness"),
    ("aging",                         "decreases",   "trabecular_connectivity"),
    ("aging",                         "decreases",   "trabecular_thickness"),
    ("aging",                         "increases",   "trabecular_spacing"),
    ("aging",                         "increases",   "microdamage_accumulation"),
    ("aging",                         "decreases",   "fracture_toughness"),
    ("aging",                         "leads_to",    "osteoporosis"),
    ("aging",                         "decreases",   "estrogen"),             # postmenopausal decline

    # ── Drug effects ──────────────────────────────────────────────────────────
    ("glucocorticoid",                "decreases",   "bone_mineral_density"),
    ("glucocorticoid",                "inhibits",    "bone_formation"),
    ("glucocorticoid",                "activates",   "osteoclast_resorption"),
    ("glucocorticoid",                "leads_to",    "osteoporosis"),
    ("bisphosphonate",                "inhibits",    "osteoclast_resorption"),
    ("bisphosphonate",                "increases",   "bone_mineral_density"),

    # ── Pathology chains ──────────────────────────────────────────────────────
    ("osteoporosis",                  "decreases",   "bone_mineral_density"),
    ("osteoporosis",                  "increases",   "porosity"),
    ("osteoporosis",                  "increases",   "fracture_risk"),
    ("osteoporosis",                  "decreases",   "trabecular_connectivity"),
    ("osteoporosis",                  "decreases",   "cortical_thickness"),
    ("osteoporosis",                  "leads_to",    "fragility_fracture"),
    ("osteoporosis",                  "leads_to",    "hip_fracture"),
    ("osteoporosis",                  "leads_to",    "vertebral_fracture"),
    ("osteopenia",                    "leads_to",    "osteoporosis"),
    ("osteogenesis_imperfecta",       "decreases",   "collagen_crosslink_density"),
    ("osteogenesis_imperfecta",       "decreases",   "fracture_toughness"),
    ("osteogenesis_imperfecta",       "leads_to",    "fragility_fracture"),
    ("Paget_disease",                 "leads_to",    "woven_bone"),
    ("Paget_disease",                 "decreases",   "elastic_modulus"),
    ("Paget_disease",                 "increases",   "fracture_risk"),
    ("hyperparathyroidism",           "activates",   "osteoclast_resorption"),
    ("hyperparathyroidism",           "decreases",   "bone_mineral_density"),
    ("hyperparathyroidism",           "increases",   "fracture_risk"),
    ("osteomalacia",                  "decreases",   "degree_of_mineralization"),
    ("osteomalacia",                  "decreases",   "elastic_modulus"),
    ("cortical_porosity_increase",    "decreases",   "elastic_modulus"),
    ("cortical_porosity_increase",    "decreases",   "fracture_toughness"),
    ("cortical_porosity_increase",    "increases",   "fracture_risk"),
    ("stress_fracture",               "leads_to",    "microdamage_accumulation"),
    ("rheumatoid_arthritis",          "activates",   "osteoclast_resorption"),
    ("rheumatoid_arthritis",          "decreases",   "bone_mineral_density"),

    # ── Clinical predictions ──────────────────────────────────────────────────
    ("bone_mineral_density",          "predicts",    "fracture_risk"),
    ("cortical_thickness",            "predicts",    "fracture_risk"),
    ("bone_strength",                 "predicts",    "fracture_risk"),
    ("FRAX_score",                    "predicts",    "fracture_risk"),
    ("fall_risk",                     "increases",   "fracture_risk"),
    ("DXA_measurement",               "measures",    "bone_mineral_density"),
    ("fracture_load",                 "determines",  "safety_factor"),
    ("safety_factor",                 "predicts",    "fracture_risk"),
    ("energy_absorption",             "determines",  "impact_resistance"),
    ("structural_rigidity",           "determines",  "load_bearing_capacity"),
    ("biomechanical_competence",      "predicts",    "fracture_risk"),

    # ── Stress shielding ──────────────────────────────────────────────────────
    ("stress_shielding",              "leads_to",    "bone_resorption"),
    ("stress_shielding",              "decreases",   "bone_mineral_density"),
    ("titanium_implant",              "leads_to",    "stress_shielding"),
    ("PEEK_implant",                  "decreases",   "stress_shielding"),
    ("periprosthetic_fracture",       "leads_to",    "fracture_risk"),

    # ── Biomaterials → bone biology ───────────────────────────────────────────
    ("bone_morphogenetic_protein",    "activates",   "bone_formation"),
    ("bone_morphogenetic_protein",    "activates",   "ossification"),
    ("demineralized_bone_matrix",     "activates",   "bone_formation"),
    ("bioactive_coating",             "activates",   "bone_formation"),
    ("bioglass",                      "activates",   "mineralisation"),

    # ── Cross-domain analogies ────────────────────────────────────────────────
    ("osteon",                        "analogous_to", "fibre_reinforced_composite"),
    ("cortical_bone",                 "analogous_to", "composite_scaffold"),
    ("trabecular_bone",               "analogous_to", "porous_scaffold"),
    ("hydroxyapatite_crystal",        "analogous_to", "hydroxyapatite_scaffold"),
    ("collagen_matrix",               "analogous_to", "fibrin_scaffold"),
    ("cement_line",                   "analogous_to", "fibre_reinforced_composite"),  # interface as deflector
]

SEED_EDGES: list[Edge] = [
    Edge(source=src, relation=rel, target=tgt, weight=1.0, evidence=[], edge_source="seed")
    for src, rel, tgt in _RAW_EDGES
]


# ── Bootstrap function ────────────────────────────────────────────────────────


def seed_graph(verbose: bool = True) -> BoneKnowledgeGraph:
    """
    Validate seed data, populate ontology.db, and return the in-memory graph.

    This function is idempotent — safe to call multiple times.  Existing
    rows are updated via upsert semantics.

    Parameters
    ----------
    verbose : bool
        If True, print progress and final stats to stdout.

    Returns
    -------
    BoneKnowledgeGraph
        The fully populated in-memory graph.
    """
    # ── 1. Build in-memory graph from seed data ────────────────────────────
    graph = BoneKnowledgeGraph()

    node_ids: set[str] = set()
    for node in SEED_NODES:
        graph.add_node(node)
        node_ids.add(node.node_id)

    skipped_edges: list[tuple[str, str, str]] = []
    for edge in SEED_EDGES:
        if edge.source not in node_ids:
            skipped_edges.append((edge.source, edge.relation, edge.target))
            continue
        if edge.target not in node_ids:
            skipped_edges.append((edge.source, edge.relation, edge.target))
            continue
        graph.add_edge(edge)

    if skipped_edges:
        logger.warning(
            "%d seed edges skipped — endpoint not in SEED_NODES:", len(skipped_edges)
        )
        for s, r, t in skipped_edges:
            logger.warning("  %s -[%s]-> %s", s, r, t)

    # ── 2. Persist to ontology.db ──────────────────────────────────────────
    with OntologyStore() as store:
        new_n, upd_n = store.upsert_nodes(list(graph.iter_nodes()))
        new_e, upd_e = store.upsert_edges(list(graph.iter_edges()))
        db_stats = store.stats()

    # ── 3. Report ──────────────────────────────────────────────────────────
    if verbose:
        g_stats = graph.stats()
        print()
        print("─" * 55)
        print("  BoneLogic — Phase 4 Step 1: Ontology bootstrap")
        print("─" * 55)
        print(f"  Nodes written   : {new_n} new · {upd_n} updated")
        print(f"  Edges written   : {new_e} new · {upd_e} updated")
        print()
        print(f"  Graph summary")
        print(f"    Total nodes   : {g_stats['n_nodes']}")
        print(f"    Total edges   : {g_stats['n_edges']}")
        print(f"    Components    : {g_stats['n_components']}")
        print(f"    Giant comp.   : {g_stats['giant_component']} nodes")
        print(f"    Density       : {g_stats['density']}")
        print()
        print("  Nodes by type")
        for ntype, count in sorted(g_stats["type_counts"].items()):
            print(f"    {ntype:<22}: {count}")
        print()
        print(f"  Database        : seed={db_stats['seed_nodes']} nodes, "
              f"{db_stats['seed_edges']} edges")
        print("─" * 55)
        print()

    return graph


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the BoneLogic bone knowledge graph from seed data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m reasoning.seed           # populate ontology.db\n"
            "  python -m reasoning.seed --stats   # show graph stats only\n"
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Load the existing ontology.db and print stats without re-seeding.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.stats:
        from reasoning.graph_db import OntologyStore

        with OntologyStore() as store:
            db_stats = store.stats()
            graph = store.load_graph()

        g_stats = graph.stats()
        print()
        print("─" * 55)
        print("  BoneLogic ontology.db — current stats")
        print("─" * 55)
        print(f"  Total nodes   : {g_stats['n_nodes']}")
        print(f"  Total edges   : {g_stats['n_edges']}")
        print(f"  Components    : {g_stats['n_components']}")
        print(f"  Giant comp.   : {g_stats['giant_component']} nodes")
        print(f"  Density       : {g_stats['density']}")
        print()
        print("  Nodes by type")
        for ntype, count in sorted(g_stats["type_counts"].items()):
            print(f"    {ntype:<22}: {count}")
        print()
        print("  Source breakdown")
        print(f"    Seed nodes    : {db_stats['seed_nodes']}")
        print(f"    Extracted     : {db_stats['extracted_nodes']}")
        print(f"    Seed edges    : {db_stats['seed_edges']}")
        print(f"    Extracted     : {db_stats['extracted_edges']}")
        print("─" * 55)
        print()
        sys.exit(0)

    seed_graph(verbose=True)


if __name__ == "__main__":
    main()
