# Project Plan: Token-Level Analysis of LLM Architectures — A Survey

**Duration:** 12 Weeks (10 weeks project + 2 weeks technical report & presentation)  
**Goal:** Produce a comprehensive survey paper reviewing how tokens are processed, transformed, and routed across four major LLM architecture families (Transformer, Mamba/SSM, Diffusion LLM, MoE), using information-theoretic, dynamical systems, and geometric perspectives to explain their performance differences.  
**Output:** Technical report + presentation for NVAITC Student Workshop by end of Week 12.

---

## Motivation & Scope

Large Language Models have diversified beyond the standard Transformer into fundamentally different architecture families — Selective State Space Models (Mamba), Diffusion Language Models (LLaDA, MDLM), and Mixture-of-Experts (MoE) — each processing tokens through radically different mechanisms:

- **Transformer:** Tokens interact via global self-attention; all-to-all pairwise computation at each layer.
- **Mamba/SSM:** Tokens are processed sequentially through a selective state space; input-dependent gating controls what information persists.
- **Diffusion LLM:** Tokens are corrupted via masking/noise and reconstructed via iterative denoising; bidirectional, non-autoregressive.
- **MoE:** Tokens are routed to specialized expert subnetworks; only a subset of parameters is activated per token.

Despite a growing body of work analyzing each architecture independently, there is no unified survey that compares *what happens to tokens* across these four families. Each community has developed its own analytical tools — information theory for Transformers, dynamical systems for Mamba, ELBO analysis for diffusion models, routing theory for MoE — with minimal cross-pollination.

**What this survey contributes:**

1. **Taxonomy:** A structured classification of token-level analysis methods organized by analytical lens (information-theoretic, dynamical systems, geometric, computational complexity) and by architecture family.
2. **Unified Token Lifecycle Framework:** A common conceptual framework describing the "lifecycle" of a token through each architecture: encoding → processing → interaction → output, with each architecture implementing these stages differently.
3. **Cross-Architecture Comparison:** Side-by-side analysis of token dynamics phenomena (clustering, collapse, dimensional reduction, information compression) across all four architectures.
4. **Gap Analysis:** Identification of missing cross-architecture analyses and unified theories that could explain *why* certain architectures outperform on specific tasks.
5. **Practical Guide:** Recommendations for practitioners on which architecture to choose based on the token-level properties their task requires.

**What this survey does NOT do:** It does not propose new architectures or new analysis methods. Original contributions are the taxonomy, the unified lifecycle framework, the cross-architecture comparison, and the gap analysis.

---

## Weeks 1–3: Literature Collection, Reading & Taxonomy Development

**Objective:** Systematically collect, read, and annotate the full body of relevant literature. Develop the taxonomy and the unified token lifecycle framework.

### Week 1: Seed Paper Collection & Database Setup

- Set up reference management (Zotero or BibTeX database).
- Collect seed papers organized by architecture and analytical lens:

**Architecture A: Transformer — Token Dynamics**

| ID | Paper | Year | Key Contribution |
|----|-------|------|-----------------|
| A1 | "Forget BIT, It is All about TOKEN" (arXiv:2511.01202) | 2025 | Token-based semantic information theory: rate-distortion, directed information, Granger causality for Transformers, Mamba, LLaDA |
| A2 | "Bridging the Dimensional Chasm" (arXiv:2503.22547) | 2025 | Expansion-contraction dynamics: tokens diffuse into high-dim then compress onto ~10-dim submanifolds |
| A3 | Massive Activations / Mix-Compress-Refine (arXiv:2510.06477) | 2025 | Attention sinks and compression valleys stem from massive activations in residual stream; 3-phase computation pattern |
| A4 | Entropy-Lens | 2025 | Shannon entropy profiles of intermediate token distributions reveal family-specific computation patterns |
| A5 | Multi-scale token clustering (arXiv:2509.25040) | 2025 | Mean-field interacting particle model: fast collapse → cluster formation → sequential merging |
| A6 | Rigollet's token collapse (arXiv:2512.01868) | 2025 | Self-attention drives tokens on $\mathbb{S}^{d-1}$ to collapse into single cluster |
| A7 | LIMe: Layer-Integrated Memory | 2025 | Per-head per-layer routing across all previous layers to prevent representation collapse |
| A8 | REFORM: Multi-Layer Aggregation | 2025 | Hierarchical hidden state integration to preserve token separability |
| A9 | Token dynamics as continuous ODE (arXiv:2502.05656) | 2025 | Transformer updates converge to ODE solution; contractive dynamics cause perturbation decay |

**Architecture B: Mamba / SSM — Token Dynamics**

| ID | Paper | Year | Key Contribution |
|----|-------|------|-----------------|
| B1 | Computational Limits of SSMs and Mamba (arXiv:2412.06148) | 2024 | Circuit complexity: Mamba and Transformers are in the same class (TC⁰) |
| B2 | Mamba Training Dynamics (arXiv:2602.12499) | 2026 | Input-dependent gating aligns with class-relevant features; non-asymptotic sample complexity bounds |
| B3 | Input Selectivity in Mamba (ICLR 2025) | 2025 | S6 layer represents Haar wavelet projections; dynamic memory decay counteraction |
| B4 | LaTIM: Token-to-Token Interactions in Mamba (ACL 2025, arXiv:2502.15612) | 2025 | Fine-grained decomposition of token interactions across layers for Mamba-1 and Mamba-2 |
| B5 | Hidden Attention in Mamba (ACL 2025) | 2025 | Mamba implements implicit attention via data-control linear operator; 3 orders of magnitude more attention matrices |

**Architecture C: Diffusion LLM — Token Dynamics**

| ID | Paper | Year | Key Contribution |
|----|-------|------|-----------------|
| C1 | LLaDA (arXiv:2502.09992) | 2025 | Masked diffusion for language: forward masking → reverse prediction; competitive with LLaMA3-8B |
| C2 | MDLM: Masked Diffusion Language Models | 2024 | Rao-Blackwellized objectives as mixtures of masked cross-entropy; tighter ELBO |
| C3 | XDLM: Unified Masked + Uniform Diffusion | 2025 | Stationary noise kernel unifying two diffusion paradigms |
| C4 | PG-DLM: Particle Gibbs for Diffusion LMs | 2025 | Trajectory-level refinement with theoretical convergence guarantees |

**Architecture D: Mixture-of-Experts — Token Routing**

| ID | Paper | Year | Key Contribution |
|----|-------|------|-----------------|
| D1 | DynaMoE: Dynamic Token-Level Expert Activation | 2026 | Variable number of active experts per token; layer-wise adaptive capacity scheduling |
| D2 | Grassmannian MoE (arXiv:2602.17798) | 2026 | Gating on Grassmannian manifold via Bingham distribution; formal bounds on routing entropy and expert collapse |
| D3 | Mixture-of-Transformers (MoT) | 2025 | Expert specialization reduces gradient conflicts; subtask strong convexity; $O(\log(\varepsilon^{-1}))$ convergence |
| D4 | DiffMoE | 2025 | Dynamic token-level routing for diffusion Transformers; batch-level global token pools |

**Architecture E: Hybrid Architectures**

| ID | Paper | Year | Key Contribution |
|----|-------|------|-----------------|
| E1 | Nemotron-3 Super | 2026 | Hybrid Mamba-Transformer MoE (120B/12B active); latent MoE + multi-token prediction |
| E2 | Jamba / Jamba-1.5 | 2024–2025 | Interleaved Mamba + Transformer + MoE layers |

- Perform forward and backward citation searches on each seed paper.
- Target: **60–80 papers** in the initial collection.

### Week 2: Extended Literature Search & Deep Reading (Batch 1)

Expand coverage using targeted queries:

| Subcategory | Search Terms | Expected Papers |
|-------------|-------------|-----------------|
| Token information content | token entropy, mutual information hidden layers, information flow transformer | 5–8 |
| Residual stream analysis | residual stream superposition, feature splitting, polysemanticity tokens | 5–8 |
| Token clustering/collapse theory | representation collapse depth, token uniformity, rank collapse | 5–8 |
| SSM expressivity | state space model expressivity, linear attention approximation, SSM vs attention | 5–8 |
| Discrete diffusion theory | discrete diffusion convergence, score matching discrete, ELBO language model | 5–8 |
| MoE load balancing | expert load balancing, auxiliary loss routing, token dropping MoE | 5–8 |
| Hybrid architecture analysis | hybrid SSM attention, Mamba-Transformer layer interleaving | 3–5 |
| Tokenization theory | BPE information theory, subword tokenization compression, vocabulary geometry | 3–5 |

- Read abstracts and introductions of all collected papers.
- Maintain a reading log: paper ID, architecture studied, analytical tool, key finding, limitations.
- Deep read the **25 most important papers** (all seed papers + key extensions). For each paper, create a structured annotation:
  1. **Architecture(s) studied:** Which of the four families? At what scale?
  2. **Token-level object:** What is measured about each token? (entropy, norm, rank, position in manifold, routing decision, masking probability)
  3. **Analytical tool:** Information theory, dynamical systems, geometry, complexity theory, empirical probing?
  4. **Layer-wise vs. global:** Does the analysis track token evolution across layers, or characterize the whole model?
  5. **Key finding about token dynamics:** What happens to tokens as they propagate through the model?
  6. **Cross-architecture implications:** Does this finding generalize to other architectures, or is it architecture-specific?
  7. **Limitations:** What architectures/scales were not tested?

### Week 3: Taxonomy & Unified Token Lifecycle Framework

**Design the taxonomy** along two axes:

**Axis 1 — Analytical Lens:**

| Lens | Core Question | Key Tools |
|------|--------------|-----------|
| Information-Theoretic | How much information does each token carry? How is it compressed/routed? | Shannon entropy, mutual information, rate-distortion, ELBO, KL-divergence |
| Dynamical Systems | How do token representations evolve as a dynamical system across layers? | ODE analysis, fixed points, Lyapunov stability, mean-field limits, contraction |
| Geometric / Manifold | What is the shape of the space tokens live in? How does it change? | Intrinsic dimension, curvature, angular dispersion, manifold capacity, clustering |
| Computational Complexity | What can/cannot be computed by token interactions in this architecture? | Circuit complexity, TC⁰, formal language recognition, counting arguments |
| Routing / Gating | How are tokens directed to different computational pathways? | Gating functions, expert assignment, load balancing, routing entropy |

**Axis 2 — Architecture Family:**

| Family | Token Processing Mechanism | Interaction Pattern |
|--------|--------------------------|-------------------|
| Transformer | Self-attention: weighted sum of all tokens | All-to-all per layer |
| Mamba/SSM | Selective state space: input-dependent recurrence | Sequential, state-mediated |
| Diffusion LLM | Iterative denoising: masked → predicted | Bidirectional, iterative |
| MoE | Expert routing: subset of parameters per token | Sparse, token-dependent |
| Hybrid | Interleaved mechanisms | Mixed |

**Design the Unified Token Lifecycle Framework:**

```
Stage 1: ENCODING
  Transformer: Embedding + Positional Encoding (sinusoidal / RoPE)
  Mamba:       Embedding + Linear Projection into state space
  Diffusion:   Embedding + Noise/Masking schedule application
  MoE:         Embedding (shared) + Initial routing decision

Stage 2: LAYER-WISE PROCESSING
  Transformer: Attention → FFN → LayerNorm (repeat L times)
  Mamba:       Conv → SSM (selective scan) → Gating (repeat L times)
  Diffusion:   Full model pass per denoising step (repeat T times)
  MoE:         Attention → Router → Expert FFN → LayerNorm (repeat L times)

Stage 3: INTER-TOKEN INTERACTION
  Transformer: Explicit via attention weights (dense, O(n²))
  Mamba:       Implicit via shared hidden state (sequential, O(n))
  Diffusion:   Implicit via bidirectional conditioning (parallel)
  MoE:         Attention (dense) + Expert (token-independent per expert)

Stage 4: OUTPUT FORMATION
  Transformer: Final LayerNorm → Linear → Softmax (autoregressive)
  Mamba:       Final Norm → Linear → Softmax (autoregressive)
  Diffusion:   Final denoising step → all tokens predicted simultaneously
  MoE:         Final expert output aggregation → Linear → Softmax
```

- Deep read remaining **15–20 papers** (Batch 2), completing annotations.

### Weeks 1–3 Deliverables

- [ ] Complete reading database with 60–80 annotated papers
- [ ] Finalized two-axis taxonomy (Analytical Lens × Architecture Family)
- [ ] Unified Token Lifecycle Framework diagram
- [ ] Literature mapping table: every paper placed in its taxonomy cell
- [ ] Preliminary paper outline

---

## Weeks 4–7: Cross-Architecture Comparison & Survey Writing

**Objective:** Perform cross-architecture analysis of token dynamics phenomena, write the main body of the survey.

### Week 4: Cross-Architecture Phenomenon Mapping

For each major token-level phenomenon identified in the literature, systematically map whether and how it manifests across all four architecture families:

| Phenomenon | Transformer | Mamba/SSM | Diffusion LLM | MoE |
|-----------|------------|-----------|---------------|-----|
| **Token clustering/collapse** | Well-studied: Rigollet's collapse theorem, angular dispersion → 0 | Unknown: does sequential processing cause clustering? | N/A (tokens corrupted, not clustered) | Partially studied: expert collapse is the analog |
| **Dimensional reduction** | Expansion-contraction: tokens compress to ~10-dim submanifolds | Unknown: does state space compress similarly? | Unknown: does denoising reduce dimension? | Unknown: do routed experts change ID? |
| **Information compression** | Mix-Compress-Refine 3-phase pattern | Selective filtering of irrelevant features | ELBO tightening across denoising steps | Expert specialization as information routing |
| **Representational rank** | Rank decreases with depth (massive activations) | Unknown | Unknown | Unknown per-expert vs. aggregate |
| **Computational complexity** | TC⁰ with polynomial precision | TC⁰ with polynomial precision (same!) | Unknown | Depends on routing mechanism |
| **Implicit attention** | Explicit by design | Hidden attention via data-control operator | Bidirectional implicit conditioning | Attention + sparse expert routing |

- For each cell marked "Unknown," note this as a gap for Section 12 (Open Problems).
- For cells with data from multiple papers, identify agreements and contradictions.

### Week 5: Literature-Based Cross-Architecture Comparison

Synthesize findings from the literature to build cross-architecture comparison tables and narratives. For each metric studied in any architecture, collect reported results and map them across families:

- Compile reported entropy, intrinsic dimension, effective rank, and cosine similarity findings from existing papers.
- Identify where literature provides direct comparisons and where only single-architecture results exist.
- Build comparison narratives based on published results, highlighting methodological differences between studies.

### Week 6: Figure Generation & Comparison Tables

- **Key figures to generate:**
  1. Token Lifecycle Framework diagram (the Stage 1–4 comparison)
  2. Taxonomy grid visualization (Analytical Lens × Architecture)
  3. Phenomenon mapping table (the "Unknown" cells highlighted as gaps)
  4. Timeline of token analysis papers (2020–2026) colored by architecture family

- **Summary comparison tables (based on literature findings):**
  | Property | Transformer | Mamba | Diffusion | MoE |
  |----------|------------|-------|-----------|-----|
  | Token interaction | Dense, O(n²) | Sequential, O(n) | Bidirectional, iterative | Sparse, token-dependent |
  | Information flow | All-to-all per layer | State-mediated | Iterative refinement | Expert-routed |
  | Collapse tendency | High (proven) | Unknown | N/A | Expert collapse |
  | Parallelism | Full (training) | Limited (inference) | Full | Full |
  | Complexity class | TC⁰ | TC⁰ | Unknown | Depends on routing |
  | Memory | O(n²) or O(n) with linear attn | O(1) per step | O(n) per step | O(n) + expert buffers |

### Week 7: Survey Body Writing

Write the main sections:

| Section | Content | Pages |
|---------|---------|-------|
| 1. Introduction | Motivation: four architectures, one question — what happens to tokens? | 1.5 |
| 2. Background | Architecture overviews, notation, token lifecycle framework | 3 |
| 3. Taxonomy | Two-axis classification with descriptions | 2 |
| 4. Information-Theoretic Token Analysis | Semantic info theory (A1), entropy profiles (A4), rate-distortion, ELBO for diffusion, routing entropy for MoE | 3.5 |
| 5. Dynamical Systems View of Token Evolution | ODE formulation (A9), mean-field clustering (A5/A6), Mamba as dynamical system (B2), contractive dynamics, denoising as dynamics | 3.5 |
| 6. Geometric Token Analysis | Dimensional reduction (A2), manifold structure, curvature, Grassmannian routing (D2), angular dispersion | 3 |
| 7. Computational Complexity | Circuit complexity (B1), expressivity separation, formal language recognition across architectures | 2.5 |
| 8. Token Routing & Gating | MoE routing theory (D1–D3), Mamba's input selectivity (B3), attention as soft routing, expert specialization | 3 |
| 9. Representation Collapse & Preservation | Collapse in Transformers (A3, A6), LIMe/REFORM solutions (A7, A8), expert collapse in MoE, unknown status in Mamba/Diffusion | 2.5 |
| 10. Hybrid Architectures | Nemotron-3 Super, Jamba: how do token dynamics change when architectures are interleaved? | 2 |
| 11. Cross-Architecture Comparison | Phenomenon mapping table, empirical comparison figures, the "big picture" | 3 |
| 12. Open Problems & Future Directions | Gap analysis | 2.5 |
| 13. Conclusion | Summary and practical recommendations | 1 |

### Weeks 4–7 Deliverables

- [ ] Cross-architecture phenomenon mapping (complete table)
- [ ] Publication-quality figures
- [ ] First draft of Sections 1–10

---

## Weeks 8–10: Gap Analysis & Finalization

**Objective:** Complete the gap analysis, write remaining sections, and finalize project work.

### Week 8: Gap Analysis & Open Problems

Systematically identify gaps from the phenomenon mapping and taxonomy:

| Gap | Description | Why It Matters |
|-----|------------|----------------|
| **Token dynamics in Mamba** | Mamba has no analog of Rigollet's collapse theorem. Does sequential processing lead to a different kind of representational degeneration? | Fundamental open question about SSM representations |
| **Diffusion LLM token geometry** | No paper studies the geometric/manifold properties of token representations during the denoising process | Would reveal whether diffusion models discover similar manifold structures to Transformers |
| **Cross-architecture token collapse theory** | Collapse is proven for Transformers but not studied for Mamba, Diffusion, or MoE. Is it a universal phenomenon or Transformer-specific? | Would determine if anti-collapse solutions (e.g., gated diversity channels, LIMe) apply beyond Transformers |
| **Unified information-theoretic framework** | Bo Bai's framework (A1) covers Transformer, Mamba, LLaDA but lacks deep comparison of information-theoretic quantities across architectures | The framework exists but cross-architecture analysis is surface-level |
| **MoE token routing geometry** | GrMoE introduces Grassmannian routing but no paper connects routing decisions to token geometry (ID, curvature) | Would explain *why* certain tokens are harder to route |
| **Hybrid architecture token lifecycle** | Nemotron-3 and Jamba interleave Mamba + Transformer + MoE, but how tokens transition between mechanisms is unstudied | Critical for designing better hybrids |
| **Token dynamics during training** | Almost all analyses study trained models. How do token dynamics evolve during pre-training? | Would connect token-level analysis to training dynamics / loss landscape theory |
| **Scale-dependent token behavior** | Most analyses at ≤7B. Do token dynamics change qualitatively at 70B+? | Determines whether findings transfer to frontier models |
| **Task-dependent token processing** | Does the same architecture process tokens differently for reasoning vs. retrieval vs. generation? | Would explain task-architecture fit |
| **Tokenizer–architecture interaction** | BPE tokenization is designed for autoregressive models. Is it optimal for Mamba? For diffusion LLMs? | Tokenizer-architecture co-design is unexplored |

- Write Section 12 (Open Problems) based on this analysis.

### Week 9: Complete First Draft & Internal Review

- Write remaining sections (11: Cross-Architecture Comparison, 13: Conclusion).
- Assemble full draft with all figures, tables, and bibliography.
- Expected bibliography: **80–120 references**.
- Internal review checklist:
  - Every claim verified against source paper
  - Phenomenon mapping table is accurate (no misattributions)
  - Taxonomy is exhaustive
  - Token Lifecycle Framework correctly represents each architecture
  - All "Unknown" cells in phenomenon mapping are reflected in Open Problems

### Week 10: Project Wrap-Up

- Finalize all project work.
- **Deliverable:** All project work complete; full manuscript draft; gap analysis.

---

## Weeks 11–12: Technical Report & Presentation

**Objective:** Write the technical report and prepare the final presentation for the NVAITC Student Workshop.

- **Technical Report Writing**
  - Revise based on internal review.
  - Add **Practitioner's Guide appendix** — a decision matrix (see Paper Structure Map).
  - Add **Architecture Quick-Reference appendix:** For each architecture, a one-page summary of: mechanism, token interaction pattern, known token dynamics, computational cost, key references.
  - Ensure all figures are colorblind-friendly with consistent color coding across architectures.
  - **Deliverable:** Complete technical report.

- **Final Presentation for NVAITC Student Workshop**
  - Prepare slides summarizing the project: motivation, four architectures, taxonomy, phenomenon mapping, gap analysis, and recommendations.
  - **Deliverable:** Presentation ready for NVAITC Student Workshop.

- **Final Formatting**
  - Final proofread and formatting (LaTeX).
  - Prepare supplementary materials:
    - Extended phenomenon mapping tables
  - **Deliverable:** Technical report + presentation.

---

## Summary: Paper Structure Map

| Section | Content | Key Tables/Figures |
|---------|---------|-------------------|
| Introduction | Four architectures, one question | Figure: architecture family tree |
| Background | Architecture overviews, notation | Figure: Token Lifecycle Framework (4-column) |
| Taxonomy | Analytical Lens × Architecture | Figure: taxonomy grid |
| Info-Theoretic Analysis | Entropy, rate-distortion, ELBO, routing entropy | Figure: entropy curves across architectures |
| Dynamical Systems View | ODE, mean-field, collapse, contraction | Figure: token trajectory phase portraits |
| Geometric Analysis | ID, curvature, manifold structure | Figure: ID across layers per architecture |
| Computational Complexity | TC⁰, expressivity, formal languages | Table: complexity class comparison |
| Token Routing & Gating | MoE routing, Mamba selectivity, attention as routing | Figure: routing decision visualization |
| Collapse & Preservation | Collapse theorems, solutions, unknowns | Figure: cosine similarity heatmaps (Transformer vs. Mamba) |
| Hybrid Architectures | Nemotron-3, Jamba | Figure: hybrid token flow diagram |
| Cross-Architecture Comparison | Phenomenon mapping | Table: phenomenon × architecture (with Unknown gaps) |
| Open Problems | Gap analysis | Table: 10 gaps with proposed methodology |
| Conclusion | Summary, recommendations | — |
| Appendix A | Practitioner's decision matrix | Table: task → architecture recommendation |
| Appendix B | Architecture quick-reference | 4 one-page summaries |

---

## Key Reference Papers (Initial Collection)

### Transformer Token Dynamics
| ID | Title | Ref | Year |
|----|-------|-----|------|
| A1 | Forget BIT, It is All about TOKEN | arXiv:2511.01202 | 2025 |
| A2 | Bridging the Dimensional Chasm | arXiv:2503.22547 | 2025 |
| A3 | Massive Activations / Mix-Compress-Refine | arXiv:2510.06477 | 2025 |
| A5 | Multi-scale Token Clustering (Mean-Field) | arXiv:2509.25040 | 2025 |
| A6 | Rigollet: Token Collapse on $\mathbb{S}^{d-1}$ | arXiv:2512.01868 | 2025 |
| A7 | LIMe: Layer-Integrated Memory | 2025 | 2025 |
| A9 | Token Dynamics as Continuous ODE | arXiv:2502.05656 | 2025 |

### Mamba / SSM Token Dynamics
| ID | Title | Ref | Year |
|----|-------|-----|------|
| B1 | Computational Limits of SSMs and Mamba | arXiv:2412.06148 | 2024 |
| B2 | Mamba Training Dynamics | arXiv:2602.12499 | 2026 |
| B3 | Input Selectivity in Mamba | ICLR 2025 | 2025 |
| B4 | LaTIM: Token Interactions in Mamba | ACL 2025 (arXiv:2502.15612) | 2025 |
| B5 | Hidden Attention in Mamba | ACL 2025 | 2025 |

### Diffusion LLM Token Dynamics
| ID | Title | Ref | Year |
|----|-------|-----|------|
| C1 | LLaDA: Diffusion Models for Language | arXiv:2502.09992 | 2025 |
| C2 | MDLM: Masked Diffusion Language Models | 2024 | 2024 |
| C3 | XDLM: Unified Masked + Uniform Diffusion | 2025 | 2025 |
| C4 | PG-DLM: Particle Gibbs for Diffusion LMs | 2025 | 2025 |

### MoE Token Routing
| ID | Title | Ref | Year |
|----|-------|-----|------|
| D1 | DynaMoE: Dynamic Token-Level Activation | 2026 | 2026 |
| D2 | Grassmannian MoE | arXiv:2602.17798 | 2026 |
| D3 | Mixture-of-Transformers: Expert Specialization | 2025 | 2025 |
| D4 | DiffMoE: Dynamic Routing for Diffusion Transformers | 2025 | 2025 |

### Hybrid Architectures
| ID | Title | Ref | Year |
|----|-------|-----|------|
| E1 | Nemotron-3 Super (Mamba-Transformer MoE) | 2026 | 2026 |
| E2 | Jamba / Jamba-1.5 | 2024–2025 | 2024 |

### Tokenization Theory
| ID | Title | Ref | Year |
|----|-------|-----|------|
| F1 | Information-Theoretic Perspective on LLM Tokenizers | arXiv:2601.09039 | 2026 |
| F2 | TokSuite: Tokenizer Choice Impact | arXiv:2512.20757 | 2025 |

---

## Timeline Summary

| Week | Phase | Activity | Deliverable |
|------|-------|----------|-------------|
| 1 | Weeks 1–3 | Seed paper collection (30+ papers), database setup | Paper database organized by architecture |
| 2 | Weeks 1–3 | Extended search, abstract reading, deep reading Batch 1 | Complete collection (70+ papers), 25 annotated papers |
| 3 | Weeks 1–3 | Taxonomy design, Token Lifecycle Framework, Batch 2 reading | **Finalized taxonomy + lifecycle framework** |
| 4 | Weeks 4–7 | Cross-architecture phenomenon mapping | **Phenomenon × Architecture table** |
| 5 | Weeks 4–7 | Literature-based cross-architecture comparison | Cross-architecture comparison from literature |
| 6 | Weeks 4–7 | Figure generation, comparison tables | **Publication figures** |
| 7 | Weeks 4–7 | Write survey body (Sections 1–10) | First draft of main sections |
| 8 | Weeks 8–10 | Gap analysis, write Sections 11–13 | **10 open problems** identified |
| 9 | Weeks 8–10 | Complete full draft, internal review | Full manuscript draft |
| 10 | Weeks 8–10 | Project wrap-up | All project work complete |
| 11 | Weeks 11–12 | Technical report writing, presentation prep | Technical report draft, presentation |
| 12 | Weeks 11–12 | Final formatting | **Technical report, presentation** |

**Exit criteria per phase:**
- **Weeks 1–3:** 70+ papers read and annotated. Taxonomy finalized. Token Lifecycle Framework validated against all four architectures.
- **Weeks 4–7:** Phenomenon mapping complete with all "Unknown" cells identified. Literature-based cross-architecture comparison complete. First draft of core sections complete.
- **Weeks 8–10:** Gap analysis identifies at least 10 concrete future directions. Full manuscript draft complete.
- **Weeks 11–12:** Technical report complete. Presentation ready for NVAITC Student Workshop.


