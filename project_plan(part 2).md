# Project Plan v2: Cross-Architecture Token Entropy Analysis and XXX Architecture Design

**Revised scope:** This plan replaces the original Week 6 onward plan. The project will no longer continue as a pure survey. Instead, it will become a research project that analyzes token entropy across four basic model architecture families and uses the findings to prepare an architecture proposal. The final architecture is intentionally left as **XXX** for the student to define after the analysis.

**Core architecture families kept in the analysis:**

1. Transformer
2. Mamba / State Space Model
3. Diffusion Language Model
4. Mixture-of-Experts

**Duration covered:** Weeks 6-12  
**Main goal:** Study how token entropy evolves in the four architecture families, then use the findings to define **XXX**, a new architecture direction to be filled in by the student.  
**Final output:** Technical report, entropy analysis results, architecture proposal, evaluation tables, and workshop presentation.

---

## 1. New Research Direction

The earlier Week 1-5 work showed that different model architectures process tokens in different ways:

- Transformers use explicit all-to-all attention.
- Mamba uses sequential state-space mixing.
- Diffusion LLMs use iterative denoising.
- MoE models use sparse expert routing.

The new project keeps these four architectures as the foundation, but changes the main question.

Original question:

```text
How do different architectures process tokens?
```

New question:

```text
Can token entropy reveal which tokens need more computation, and can that signal guide a new architecture?
```

The project has two parts:

1. **Analysis:** Measure token entropy across Transformer, Mamba, Diffusion LLM, and MoE models.
2. **Architecture:** Use the entropy patterns to define **XXX**, with the final design left for the student to fill in.

---

## 2. Central Hypothesis

Token entropy is not random noise. It reflects the computational state of each token.

Low-entropy tokens are usually already stable. High-entropy tokens often need more context, more reasoning, or more refinement. If this pattern is reliable across architectures, entropy can be used as a practical control signal.

**Main hypothesis:**

```text
Token entropy can predict token difficulty, token stability, and the amount of computation a token should receive.
```

**Architecture hypothesis for XXX:**

```text
XXX may improve the quality-speed tradeoff if it uses token entropy to allocate computation more selectively than fixed-compute models.
```

---

## 3. What Will Be Measured

The project will measure entropy at the token level.

For a token position `i`, entropy is:

```text
H_i = - sum_v p_i(v) log p_i(v)
```

where:

- `i` is the token position
- `v` is a vocabulary item
- `p_i(v)` is the model probability for vocabulary item `v` at position `i`

For architectures with multiple layers or denoising steps, entropy becomes:

```text
H_i^t = - sum_v p_i^t(v) log p_i^t(v)
```

where `t` can mean layer index, Mamba block index, diffusion step, or MoE layer.

### 3.1 Shared Token Signals

These signals will be measured for all four architectures where possible:

| Signal | Meaning | Why It Matters |
|--------|---------|----------------|
| Token entropy | Prediction uncertainty | Finds easy vs hard tokens |
| Entropy drop | Entropy change across depth or steps | Shows whether a token is resolving |
| Top-token confidence | Probability of most likely token | Simple stability signal |
| Token change rate | Whether predicted token changes over time | Tests whether a token can be frozen |
| Final correctness | Whether the final token prediction is correct | Links entropy to output quality |
| Effective rank | Rank of token representation matrix | Detects compression or collapse |
| Cosine similarity | Similarity between token representations | Detects token clustering |

---

## 4. Architecture-Specific Analysis

The key part of the project is to keep the four basic architectures separate during analysis before proposing a new design.

### 4.1 Transformer Token Entropy

Transformers process tokens using dense self-attention.

Measure:

- entropy across layers
- attention entropy by head and layer
- residual-stream prediction entropy
- entropy drop from early to late layers
- effective rank of hidden states
- cosine similarity between token representations

Questions:

- Do tokens become more certain with depth?
- Are high-attention tokens also low-entropy tokens?
- Do important tokens stay high-entropy longer?
- Does entropy drop align with representational compression?

Expected output:

- layer-by-layer entropy curves
- attention entropy vs prediction entropy table
- token collapse or compression plots

### 4.2 Mamba / SSM Token Entropy

Mamba processes tokens through selective state-space recurrence.

Measure:

- entropy across Mamba blocks
- hidden-state change across layers
- token stability across sequence position
- entropy behavior for early vs late tokens
- effective rank of hidden states
- implicit token interaction if LaTIM-style analysis is available

Questions:

- Does Mamba show the same entropy reduction pattern as Transformers?
- Are long-distance dependency tokens higher entropy?
- Does the state-space mechanism compress tokens differently from attention?
- Do early tokens behave differently because later tokens depend on state memory?

Expected output:

- block-wise entropy curves
- entropy by token position
- comparison with Transformer entropy curves

### 4.3 Diffusion LLM Token Entropy

Diffusion LLMs process tokens through iterative denoising.

Measure:

- entropy across denoising steps
- entropy drop per token
- token prediction changes across denoising steps
- mask reconstruction accuracy
- final correctness vs early entropy

Questions:

- Which tokens become certain early?
- Which tokens remain uncertain until late denoising steps?
- Can low-entropy tokens be frozen safely?
- Do answer-critical tokens require more denoising steps?

Expected output:

- denoising-step entropy curves
- token-freezing safety table
- entropy vs final reconstruction accuracy

### 4.4 MoE Token Entropy

MoE models route tokens to experts.

Measure:

- prediction entropy
- router entropy
- top expert probability
- number of active experts per token
- expert load balance
- expert choice stability across layers
- final correctness vs routing entropy

Questions:

- Are high-entropy tokens routed to more specialized experts?
- Does router entropy predict token difficulty?
- Do unstable tokens switch experts more often?
- Is expert collapse related to low routing entropy?

Expected output:

- routing entropy curves
- expert assignment stability table
- token prediction entropy vs expert entropy analysis

---

## 5. Cross-Architecture Comparison

After the four separate analyses, compare them using one shared table.

| Question | Transformer | Mamba | Diffusion LLM | MoE |
|----------|-------------|-------|---------------|-----|
| Where does token entropy decrease? | Across layers | Across blocks/state updates | Across denoising steps | Across layers and expert choices |
| What is the token interaction mechanism? | Attention | State-space recurrence | Bidirectional denoising | Expert routing plus backbone attention |
| What counts as token stability? | Same prediction across layers | Same prediction across blocks | Same prediction across denoising steps | Same prediction and same expert route |
| What is the natural compute-control signal? | Attention/prediction entropy | State change/entropy | Denoising entropy | Router entropy and prediction entropy |
| What token gets extra computation? | High entropy or high attention | High entropy or high state change | High entropy or unstable token | High prediction entropy or high routing entropy |

The goal is to identify which entropy behaviors are universal and which are architecture-specific.

---

## 6. New Architecture Direction: XXX

The new architecture is intentionally left as **XXX**.

The student should not decide the final architecture before the entropy analysis is complete. The purpose of Weeks 6-9 is to discover which token-entropy patterns are real. Only after that should the student fill in **XXX**.

### 6.1 What XXX Must Be Based On

XXX should be designed from the cross-architecture findings:

| Finding Type | How It Should Influence XXX |
|--------------|-----------------------------|
| Transformer entropy pattern | Decide whether global attention is needed for high-entropy tokens |
| Mamba entropy pattern | Decide whether state-space computation is enough for stable or local tokens |
| Diffusion entropy pattern | Decide whether iterative refinement is needed for unresolved tokens |
| MoE routing entropy pattern | Decide whether uncertain tokens benefit from expert specialization |
| Token stability pattern | Decide whether low-entropy tokens can be skipped or cheaply updated |

### 6.2 Student Fill-In Section

The student should fill in the following after Week 9:

```text
Architecture name: XXX

Main idea:
XXX

Architecture components:
1. XXX
2. XXX
3. XXX
4. XXX

Token entropy role:
XXX

How Transformer analysis influences the design:
XXX

How Mamba analysis influences the design:
XXX

How Diffusion analysis influences the design:
XXX

How MoE analysis influences the design:
XXX

Expected advantage:
XXX
```

### 6.3 Candidate Design Questions

The student should answer these before finalizing XXX:

1. Should XXX use one backbone or multiple compute paths?
2. Should entropy control token routing, token freezing, denoising steps, expert choice, or attention selection?
3. Should the entropy rule be threshold-based or learned?
4. Should XXX focus on speed, accuracy, calibration, long-context ability, or reasoning?
5. Which architecture family contributes the most useful mechanism?

### 6.4 Placeholder Routing Rule

The routing rule is not finalized. Keep it as:

```text
XXX routing rule:
XXX
```

---

## 7. Datasets

The dataset choice must support both token-level analysis and output-quality evaluation.

### 7.1 Required Dataset 1: WikiText-103

Purpose:

- main controlled language modeling dataset
- token entropy analysis
- reconstruction and prediction quality

Why use it:

- clean text
- standard benchmark
- manageable size
- good for token-level plots

Metrics:

- negative log likelihood
- bits per token
- masked token accuracy
- entropy vs correctness
- entropy vs token type

### 7.2 Required Dataset 2: GSM8K

Purpose:

- reasoning evaluation
- test whether high-entropy tokens correspond to reasoning-critical positions

Why use it:

- clear final answer
- common reasoning benchmark
- easy output judgment

Metrics:

- exact-match final answer
- entropy on numbers and answer tokens
- entropy on reasoning-step tokens
- quality-speed tradeoff

### 7.3 Required Dataset 3: ARC-Challenge

Purpose:

- knowledge and reasoning robustness
- multiple-choice output judgment

Why use it:

- clear correct answer
- harder than simple language modeling
- useful for comparing uncertainty and correctness

Metrics:

- multiple-choice accuracy
- entropy of answer options
- calibration error

### 7.4 Optional Dataset 4: PG-19

Purpose:

- long-context generation
- test Mamba and Transformer differences

Why use it:

- long documents
- useful for state memory and attention analysis

Metrics:

- negative log likelihood
- entropy by position
- long-range dependency examples

### 7.5 Optional Dataset 5: LongBench Subset

Purpose:

- long-context question answering
- test whether high-entropy global tokens benefit from attention path

Metrics:

- exact match
- F1
- entropy of evidence tokens
- performance vs context length

### 7.6 Optional Synthetic Tasks

Use synthetic tasks only if real-data analysis is unclear.

Suggested tasks:

- copy task
- key-value retrieval
- bracket matching
- passkey retrieval
- simple arithmetic chains

Why use them:

- ground truth is exact
- token difficulty can be controlled
- good for debugging entropy routing

---

## 8. Baselines

The project should compare against basic versions of the four architecture families.

| Baseline | Purpose |
|----------|---------|
| Small Transformer LM | Basic attention baseline |
| Small Mamba LM | State-space baseline |
| Small Diffusion LM or masked diffusion prototype | Iterative denoising baseline |
| Small MoE LM or router-instrumented model | Expert routing baseline |
| XXX prototype | Student-defined architecture placeholder |

If full training is too expensive, the analysis can use available pretrained small models and a lightweight prototype for the entropy-routing experiment.

---

## 9. How to Judge the Output Clearly

The project must avoid vague judgments like "the text looks better." Each output should be judged with explicit metrics.

### 9.1 Language Modeling Metrics

| Metric | Meaning | Better Direction |
|--------|---------|------------------|
| Negative log likelihood | Probability assigned to correct tokens | Lower |
| Bits per token | Compression quality | Lower |
| Masked token accuracy | Correct reconstruction of masked tokens | Higher |
| Per-token entropy | Uncertainty of each token | Interpreted with correctness |

### 9.2 Task Metrics

| Dataset | Metric |
|---------|--------|
| WikiText-103 | NLL, bits per token, masked token accuracy |
| GSM8K | exact-match final answer |
| ARC-Challenge | multiple-choice accuracy |
| PG-19 | NLL and entropy by position |
| LongBench subset | exact match or F1 |
| Synthetic tasks | exact match |

### 9.3 Entropy Quality Metrics

These metrics judge whether entropy is useful.

| Metric | Question | Target |
|--------|----------|--------|
| Entropy-error AUROC | Does entropy predict wrong tokens? | Higher than 0.75 |
| Entropy-change AUROC | Does entropy predict future token changes? | Higher than 0.75 |
| Freeze precision | Are frozen tokens actually stable? | Higher than 95% |
| Freeze recall | How many stable tokens are found? | Higher is better |
| Calibration error | Does confidence match correctness? | Lower than baseline |

### 9.4 Architecture Metrics

These metrics judge whether XXX is useful after the student defines it.

| Metric | Meaning | Better Direction |
|--------|---------|------------------|
| Tokens per second | Inference speed | Higher |
| Average compute paths per token | Cost of adaptive routing | Lower |
| Percent frozen tokens | Computation skipped | Higher if quality remains stable |
| GPU memory | Runtime memory use | Lower |
| Quality-speed Pareto curve | Tradeoff between quality and efficiency | Better curve than baselines |

### 9.5 Clear Success Criteria

The project succeeds if at least two of the following hold:

1. Token entropy predicts final token errors with AUROC above 0.75.
2. Low-entropy token freezing keeps quality within 1-2% of the non-freezing baseline.
3. XXX improves inference speed by at least 20% at similar quality.
4. XXX improves quality at the same compute budget.
5. The analysis finds a clear difference between Transformer, Mamba, Diffusion, and MoE entropy behavior.

---

## 10. Week-by-Week Plan

## Week 6: Build Cross-Architecture Entropy Pipeline

**Objective:** Prepare a shared token entropy measurement pipeline for all four architecture families.

Tasks:

- Choose small model candidates for:
  - Transformer
  - Mamba
  - Diffusion LLM or masked diffusion prototype
  - MoE
- Prepare WikiText-103 samples.
- Implement shared entropy extraction.
- Save per-token records:
  - token text
  - token position
  - entropy
  - top-token confidence
  - predicted token
  - final correctness
  - layer, block, denoising step, or MoE layer

Deliverables:

- shared entropy extraction script
- first entropy table
- first entropy plots for at least two architectures

---

## Week 7: Transformer and Mamba Entropy Analysis

**Objective:** Compare attention-based and state-space token entropy behavior.

Tasks:

- Run Transformer entropy analysis.
- Run Mamba entropy analysis.
- Compare entropy across layers or blocks.
- Measure effective rank and cosine similarity if possible.
- Identify tokens that stay high entropy.

Deliverables:

- Transformer entropy curve
- Mamba entropy curve
- comparison table
- short memo: "Does Mamba reduce token entropy like a Transformer?"

---

## Week 8: Diffusion and MoE Entropy Analysis

**Objective:** Add denoising and expert-routing models to the comparison.

Tasks:

- Run diffusion token entropy analysis across denoising steps.
- Track token prediction changes across steps.
- Run MoE entropy and router entropy analysis.
- Compare prediction entropy with routing entropy.
- Build the first four-architecture comparison table.

Deliverables:

- diffusion entropy curve
- MoE prediction entropy and router entropy plots
- cross-architecture entropy comparison table

---

## Week 9: Token Stability and Compute-Control Study

**Objective:** Test whether entropy can decide token compute allocation.

Tasks:

- Define stability rules:
  - low entropy
  - high confidence
  - prediction unchanged for k steps
  - entropy drop below threshold
- Test freezing or skipping low-entropy tokens.
- Test extra refinement for high-entropy tokens.
- Measure freeze precision and recall.
- Evaluate quality loss from skipping stable tokens.

Deliverables:

- token-freezing table
- quality vs skipped-compute plot
- answer to whether entropy is a useful control signal

---

## Week 10: XXX Architecture Design

**Objective:** Fill in XXX based on the token entropy analysis from Weeks 6-9.

Tasks:

- Define XXX:
  - architecture name
  - main idea
  - components
  - token entropy role
  - relationship to Transformer analysis
  - relationship to Mamba analysis
  - relationship to Diffusion analysis
  - relationship to MoE analysis
- Write routing equations.
- Draw architecture diagram.
- Decide first prototype scope:
  - simple hard routing
  - Mamba plus attention path
  - Mamba plus MoE path
  - diffusion-style refinement loop

Deliverables:

- XXX architecture specification
- architecture diagram
- routing rule
- prototype plan

---

## Week 11: Prototype and Evaluation

**Objective:** Build a minimal prototype or simulation of entropy-guided routing.

Tasks:

- Implement simple entropy routing.
- Compare against fixed-compute baseline.
- Evaluate on WikiText-103.
- Evaluate on GSM8K and ARC-Challenge if possible.
- Run ablations:
  - confidence-only routing
  - entropy-only routing
  - entropy plus token-change routing
  - with and without freezing

Deliverables:

- prototype result table
- ablation table
- speed and quality comparison
- failure case examples

---

## Week 12: Final Report and Presentation

**Objective:** Prepare final workshop materials.

Report structure:

| Section | Content |
|---------|---------|
| 1. Introduction | Why token entropy matters |
| 2. Background | Transformer, Mamba, Diffusion LLM, MoE |
| 3. Method | Shared token entropy measurement |
| 4. Cross-Architecture Analysis | Entropy behavior in the four families |
| 5. Stability Study | Whether entropy predicts token stability |
| 6. XXX | Student-defined architecture |
| 7. Experiments | Datasets, baselines, metrics |
| 8. Results | Quality, speed, calibration, routing |
| 9. Limitations | Compute limits and model availability |
| 10. Conclusion | Whether entropy can guide architecture design |

Deliverables:

- final technical report
- final slide deck
- main architecture diagram
- cross-architecture entropy table
- final evaluation table

---

## 11. Expected Contributions

This project should produce four contributions:

1. **Cross-architecture token entropy analysis**
   - Compares Transformer, Mamba, Diffusion LLM, and MoE using the same entropy framework.

2. **Token stability study**
   - Tests whether low-entropy tokens can be safely skipped, frozen, or cheaply updated.

3. **Entropy-guided architecture proposal**
   - Defines XXX, where the student fills in how token entropy should affect architecture design.

4. **Clear evaluation framework**
   - Judges the model using quality, efficiency, calibration, and entropy-routing metrics.

---

## 12. Risks and Backup Plans

| Risk | Backup Plan |
|------|-------------|
| Some architecture models are hard to run | Use small models or simplified prototypes |
| Diffusion LLM code is unavailable | Use a masked-denoising prototype |
| MoE model is too large | Use a small router-instrumented MoE or simulate routing |
| Entropy does not predict correctness | Test entropy drop, confidence, and token-change rate |
| XXX prototype is too hard | Deliver architecture design plus routing simulation |
| Compute is limited | Focus on WikiText-103, GSM8K, and ARC-Challenge |

---

## 13. Final Research Claim

The final report should answer:

```text
Can token entropy provide a shared analysis tool across Transformer, Mamba, Diffusion LLM, and MoE models, and can it guide the design of a new adaptive architecture?
```

If yes, the project proposes XXX as a new direction to be completed by the student.

If no, the project still contributes by showing where entropy is useful, where it fails, and which architecture families need different token-level signals.
