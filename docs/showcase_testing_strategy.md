# Showcase & Stress-Testing Disease Selection

This document outlines the disease selection and testing strategy for demonstrating and stress-testing `clinical-graph`.

## Alignment with the `clinical-graph` Schema

In [`extractor.py`](../src/clinical_graph/agents/extractor.py), the two-stage `GraphExtractor` extracts:
- **Node types:** `drugs`, `conditions`, `adverse effects`, `populations`, `biomarkers`, and `procedures` (with `aliases` and descriptions embedded in `LocalVectorStore`).
- **Edge types:** Directed clinical relationships (`TREATS`, `CAUSES`, `CONTRAINDICATED_IN`, `INTERACTS_WITH`, `REQUIRES_MONITORING`) backed by verbatim source-text quotes.

The query pipeline combines vector similarity search over node descriptions with configurable $k$-hop (`graph_k_hops`) directed graph traversal (see [architecture.md](./architecture.md)). An effective showcase needs diseases that exercise all of these node and edge types at varying levels of graph density and ambiguity.

---

## Tier 1: Primary Showcase — Type 2 Diabetes (T2D)

**Role:** Baseline ("happy path") demonstration for ingestion, alias resolution, and 1–2 hop clinical retrieval.

### Rationale
- **Full schema coverage at manageable complexity:** T2D naturally spans drugs (Metformin, SGLT2 inhibitors, GLP-1 receptor agonists, sulfonylureas, insulin), biomarkers (`HbA1c`, `eGFR`, `UACR`), comorbidities/conditions (CKD, heart failure, ASCVD), adverse effects (lactic acidosis, hypoglycemia, GI intolerance), and monitoring procedures.
- **Clear contraindication and monitoring edges:** Rules such as `Metformin` `CONTRAINDICATED_IN` severe renal impairment (`eGFR < 30`) or `REQUIRES_MONITORING` for renal function and vitamin B12 produce unambiguous, verifiable verbatim-quote edges.
- **Intuitive multi-hop queries:** Relatable to both technical and clinical audiences (e.g., *"Which diabetes drug improves heart failure outcomes and what renal monitoring does it require?"*).
- **Accessible reference guidelines:** Openly available ADA/EASD consensus guidelines make ground-truth verification straightforward.

---

## Tier 2: Alternative & Complementary Baseline Candidates

Depending on which capability of `clinical-graph` is being highlighted, the following conditions offer similar structural clarity to T2D:

| Condition | Primary Showcase Strength | Key Graph Patterns |
| --- | --- | --- |
| **Heart Failure (HFrEF / HFpEF)** | Guideline-directed 4-pillar therapy & alias matching | Tight `REQUIRES_MONITORING` (`potassium`, `creatinine`, `BP`), washout `CONTRAINDICATED_IN` rules (ACEi ↔ ARNI), brand/generic aliases (`sacubitril/valsartan` / `Entresto`). |
| **Atrial Fibrillation (AFib) & Anticoagulation** | Safety-critical edge density (`clinical-guard` & Audit Agent) | Heavy `INTERACTS_WITH` (DOACs vs. Warfarin, CYP3A4/P-gp inhibitors, NSAIDs), renal dosing thresholds, `CONTRAINDICATED_IN` (mechanical valves). |
| **Rheumatoid Arthritis (RA)** | Pre-treatment screening & monitoring procedures | Step-up csDMARD → biologic/JAKi ladders; baseline latent TB / Hepatitis B `procedures` before TNF inhibitors; CBC/LFT monitoring on methotrexate. |
| **Asthma / COPD** | Step-wise escalation & biomarker-gated biologics | GINA/GOLD steps, `blood eosinophils` / `FeNO` / `IgE` biomarkers, drug–condition contraindications (non-selective beta-blockers). |

---

## Tier 3: Oncology Stress Test — Non-Small Cell Lung Cancer (NSCLC)

**Role:** High-complexity stress test for entity disambiguation, multi-hop traversal, and context-sufficiency fallback.

Broad "oncology" is too diffuse for a single coherent test corpus; selecting a single biomarker-driven malignancy provides a reproducible stress test. **NSCLC** is the primary stress-test target.

### Rationale
- **Granular biomarker stratification:** Requires distinguishing closely related molecular subtypes (`EGFR` exon 19 deletion, L858R, T790M; `ALK`; `ROS1`; `KRAS G12C`; `PD-L1` TPS) in both node extraction and vector similarity search.
- **Heavy alias normalization:** Targeted therapies carry multiple identifiers (generic, brand, investigational codes such as `osimertinib` / `Tagrisso` / `AZD9291`), testing alias deduplication and canonicalization.
- **Deep multi-hop resistance chains:** Queries naturally span 2–3+ hops: *biomarker → 1st-line TKI → resistance mutation → 2nd-line therapy → toxicity monitoring*.
- **Distinct toxicity & monitoring profiles:** Contrasts immune-related adverse events from checkpoint inhibitors (pneumonitis, colitis, thyroiditis) against TKI-specific risks (interstitial lung disease, QTc prolongation).
- **Fallback & audit triggering:** Rapidly evolving or niche mutation combinations naturally test the "Sufficient Local Context?" decision boundary and planned web-search fallback / Audit Agent behavior.

### Alternative Oncology Stress-Test Picks
- **HER2-Positive / Triple-Negative Breast Cancer:** Best for testing combination regimens, receptor/population branching (`ER`/`PR`/`HER2`, `BRCA1/2`, menopausal status), and organ-specific monitoring (`trastuzumab` + `LVEF` echocardiogram monitoring).
- **Chronic Myeloid Leukemia (CML):** Smaller corpus than NSCLC, but stresses mutation-driven TKI switching (`BCR-ABL1` T315I → ponatinib/asciminib) and cardiovascular/drug-interaction trade-offs.

---

## Recommended Showcase Progression

1. **Start with Type 2 Diabetes** to validate end-to-end ingestion, clean graph visualization, alias lookup, and 1–2 hop grounded Q&A.
2. **Optionally add AFib/Anticoagulation** when demonstrating safety guardrails (`clinical-guard`) and `INTERACTS_WITH` / `CONTRAINDICATED_IN` verification.
3. **Run NSCLC as the stress test** to benchmark `GraphExtractor` precision, $k$-hop traversal depth (`graph_k_hops`), and fallback behavior on dense, biomarker-stratified clinical text.
