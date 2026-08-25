# Token Lifecycle and Entropy Analysis Across LLM Architectures

This repository contains the code and final reports for our research conducted under Nvidia's MATH4AI programme. Our study provides a comprehensive comparative analysis of token dynamics—specifically focusing on token entropy and lifecycle stages—across major Large Language Model (LLM) architectures.

## Unified Token Lifecycle Framework
To facilitate cross-architectural comparison, the accompanying report introduces a framework that maps token processing into five distinct stages:
*   **Initialization & Conditioning**
*   **Sequence & Spatial Mixing**
*   **Transformation & Conditional Routing**
*   **Depth & Temporal Evolution**
*   **Projection & Emission**

## Supported Architectures
We evaluate token dynamics across four primary architectural families through this lens:
*   **Transformers:** Characterized by explicit, all-to-all dense interactions and continuous spatial depth, where geometric forces naturally lead to phenomena like token collapse and "Attention Sinks."
*   **Mamba / State Space Models (SSMs):** Defined by sequential recurrence and input-dependent selectivity, which allow the model to efficiently filter and process tokens through "hidden" attention mechanisms.
*   **Diffusion LLMs:** Departs from standard left-to-right processing by utilizing bidirectional context and iterative temporal denoising to simultaneously refine token certainty.
*   **Mixture-of-Experts (MoE):** Distinguished by conditional routing, dynamically assigning specialized computational experts based on an individual token's complexity gradient and semantic category.

## Entropy Analysis and Codebase
The second part of the research focused on analyzing entropy of different architectures. Models in this study were trained and tested on the Wikitext dataset (unless otherwise stated). 
*   **General Algorithm:** We have provided a general algorithm in this repository so that users can easily analyze the token entropy of any open-weight LLMs of their choice on any dataset of their choice.

## Key Findings during Entropy Analysis
Key insights from the entropy analysis include:
*   **Family Clustering:** Entropy profiles cluster strictly by model family rather than parameter size, indicating shared prediction dynamics driven by architectural and training choices.
*   **Depth-Rescaling Invariance:** Models within the same family show striking trajectory alignments when mapped to relative layer depth, implying that model scale dictates computational resolution rather than qualitative for.
*   **Diffusion "Ignorance is Bliss" Dynamics:** Diffusion models exhibit a non-linear entropy curve starting at zero static, peaking into high uncertainty as signals emerge, and dropping back to zero upon crystallization.
*   **MoE Router Independence:** In Mixture-of-Experts models, router entropy (expert uncertainty) remains mathematically decoupled and independent from token prediction entropy.
