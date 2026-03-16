"""
Bone-domain keyword taxonomy for Semantic Scholar queries.

Organised by sub-domain so queries can be run selectively or all at once.
Each entry is a search string passed directly to the S2 /paper/search endpoint.
"""

KEYWORD_GROUPS: dict[str, list[str]] = {
    # ── Bone types: skull & cranium ──────────────────────────────────────────
    "bone_types_skull": [
        "frontal bone skull",
        "parietal bone cranium",
        "temporal bone anatomy",
        "occipital bone structure",
        "sphenoid bone",
        "ethmoid bone",
        # facial bones
        "nasal bone fracture",
        "maxilla bone structure",
        "zygomatic bone cheekbone",
        "lacrimal bone",
        "palatine bone",
        "inferior nasal concha",
        "vomer bone",
        "mandible bone mechanics",
    ],
    # ── Bone types: ear ossicles ─────────────────────────────────────────────
    "bone_types_ear": [
        "malleus incus stapes ossicles",
        "middle ear ossicle mechanics",
        "ossicle bone microstructure",
    ],
    # ── Bone types: hyoid & throat ───────────────────────────────────────────
    "bone_types_hyoid": [
        "hyoid bone fracture",
        "hyoid bone anatomy mechanics",
    ],
    # ── Bone types: spine & vertebrae ───────────────────────────────────────
    "bone_types_spine": [
        "cervical vertebra bone mechanics",
        "atlas axis C1 C2 vertebra",
        "thoracic vertebra spine",
        "lumbar vertebra bone structure",
        "sacrum bone pelvis",
        "coccyx bone anatomy",
        "vertebral body trabecular bone",
        "intervertebral disc vertebra",
        "vertebral compression fracture",
        "spine bone mineral density",
    ],
    # ── Bone types: thoracic cage ────────────────────────────────────────────
    "bone_types_thorax": [
        "rib bone fracture mechanics",
        "rib cortical bone structure",
        "sternum bone anatomy",
        "clavicle bone fracture",
        "scapula bone mechanics",
    ],
    # ── Bone types: upper limb ───────────────────────────────────────────────
    "bone_types_upper_limb": [
        "humerus bone fracture mechanics",
        "radius bone wrist fracture",
        "ulna bone forearm",
        "carpal bone wrist scaphoid",
        "scaphoid bone fracture",
        "lunate bone avascular necrosis",
        "metacarpal bone hand",
        "phalanx finger bone",
    ],
    # ── Bone types: pelvis & hip ─────────────────────────────────────────────
    "bone_types_pelvis": [
        "pelvis hip bone anatomy",
        "ilium bone mechanics",
        "ischium bone structure",
        "pubis bone symphysis",
        "acetabulum hip joint bone",
        "femoral head bone necrosis",
    ],
    # ── Bone types: lower limb ───────────────────────────────────────────────
    "bone_types_lower_limb": [
        "femur bone mechanical properties",
        "femoral neck fracture osteoporosis",
        "proximal femur bone density",
        "tibia bone fracture mechanics",
        "tibial plateau bone structure",
        "fibula bone anatomy",
        "patella bone kneecap",
        "calcaneus heel bone mechanics",
        "talus bone ankle",
        "navicular bone foot",
        "metatarsal bone foot fracture",
        "toe phalanx bone",
    ],
    # ── Bone types: by tissue category ──────────────────────────────────────
    "bone_types_tissue": [
        "long bone diaphysis cortical",
        "short bone carpal tarsal",
        "flat bone skull plate",
        "irregular bone vertebra",
        "sesamoid bone patella",
        "cancellous trabecular bone",
        "compact cortical bone",
        "periosteal bone formation",
        "endochondral ossification bone",
        "intramembranous ossification bone",
    ],
    # ── Morphology & anatomy ────────────────────────────────────────────────
    "morphology": [
        "bone morphology",
        "cortical bone microstructure",
        "trabecular bone architecture",
        "osteocyte lacunar network",
        "haversian canal osteon",
        "bone remodeling morphology",
        "periosteum endosteum structure",
        "bone marrow niche",
    ],
    # ── Structure-function ───────────────────────────────────────────────────
    "structure_function": [
        "bone structure function relationship",
        "bone hierarchical structure",
        "collagen mineral composite bone",
        "hydroxyapatite collagen bone",
        "bone anisotropy mechanical",
        "lamellar bone structure",
        "woven bone primary bone",
        "bone poroelasticity",
    ],
    # ── Mechanics ────────────────────────────────────────────────────────────
    "mechanics": [
        "bone mechanical properties",
        "cortical bone fracture toughness",
        "trabecular bone compressive strength",
        "bone fatigue failure",
        "bone viscoelasticity",
        "bone elastic modulus",
        "bone crack propagation",
        "bone nanoindentation",
        "bone finite element analysis",
        "bone yield strength",
    ],
    # ── Pathology ────────────────────────────────────────────────────────────
    "pathology": [
        "osteoporosis bone loss",
        "osteogenesis imperfecta",
        "Paget disease bone",
        "bone metastasis",
        "stress fracture bone",
        "osteonecrosis avascular necrosis",
        "osteoarthritis subchondral bone",
        "rickets osteomalacia",
        "bone dysplasia",
        "osteosarcoma",
    ],
    # ── Imaging ──────────────────────────────────────────────────────────────
    "imaging": [
        "bone X-ray radiograph analysis",
        "bone MRI imaging",
        "microCT bone imaging",
        "DXA bone mineral density",
        "quantitative CT bone",
        "bone ultrasound assessment",
        "bone DEXA scan",
    ],
    # ── Biomaterials & bio-inspired ──────────────────────────────────────────
    "biomaterials": [
        "bone biomaterial scaffold",
        "bone tissue engineering",
        "bio-inspired bone material",
        "bone substitute graft",
        "hydroxyapatite scaffold",
        "bone cement polymer composite",
        "3D printing bone scaffold",
        "bone regeneration biomaterial",
    ],
    # ── Simulation & modelling ───────────────────────────────────────────────
    "simulation": [
        "bone computational modelling",
        "bone finite element simulation",
        "bone multiscale modelling",
        "bone remodelling mathematical model",
        "molecular dynamics bone mineral",
        "bone mechanobiology simulation",
    ],
    # ── Cell biology ─────────────────────────────────────────────────────────
    "cell_biology": [
        "osteoblast osteoclast bone",
        "bone cell signalling",
        "RANKL RANK OPG bone",
        "osteocyte mechanosensing",
        "bone marrow mesenchymal stem cell",
        "bone wnt signalling pathway",
    ],
}

# Flat list for convenience
ALL_KEYWORDS: list[str] = [kw for group in KEYWORD_GROUPS.values() for kw in group]
