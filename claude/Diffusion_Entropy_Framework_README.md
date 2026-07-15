# Diffusion-LLM Entropy Framework — Analysis & Corrected Extraction

## 1. Which heatmap is the "correct representation"?

**Both** notebooks draw the *same* heatmap, and the representation is a reasonable one:

```python
heatmap_data = df.pivot(index='canvas_position',
                        columns='denoising_step',
                        values='prediction_entropy')
sns.heatmap(heatmap_data, cmap="viridis", ...)
```

Rows = token position in the generated block, columns = denoising step, colour =
per-token vocabulary entropy. It is a **legitimate but not canonical** way to
visualise how a diffusion LM's uncertainty collapses over its denoising trajectory,
and it matches the project plan's "entropy across denoising steps / entropy drop
per token" goal. **Key caveat:** both notebooks take entropy over *every* canvas
position at *every* forward pass — including positions already committed, which
re-encode a known token at ~0 entropy — so the picture largely encodes **commit
order**, not pure per-token difficulty. The mask-aware variant in this rewrite is
what fixes that.

Between the two, the **`Framework (Diffusion).ipynb` heatmap is the better of the
two**, but for a narrower reason than it looks:
- The heatmaps render **only** `prediction_entropy`, so the MoE-router change
  (optional vs. mandatory) has **no effect on the heatmap** — it only feeds the
  separate router scatter plot. Router-optionality matters for the framework's
  *generality*, not for this image.
- The single heatmap-relevant code difference is the Framework extractor's
  `prediction_logits.pop(0)`, which drops the prompt-evaluation ("prefill") pass so
  step 0 is the first real denoising step. (That `pop(0)` is itself model-specific
  and is actually *wrong* for reference LLaDA/Dream, which have no prefill pass —
  see §2, row 5.)
- The two rendered images also differ simply because they were run on **different
  WikiText inputs/prompts**, so their values and step counts aren't directly
  comparable.

So **neither extraction is correct or general as written.** The heatmap *shape*
is fine; the *data pipeline* behind it is what needs fixing.

## 2. What is wrong with the extraction (not the picture)

| # | Problem | Where | Why it's wrong |
|---|---------|-------|----------------|
| 1 | Fictional model `DiffusionGemmaForBlockDiffusion` / `google/diffusiongemma-26B-A4B-it` | both | No such class or checkpoint exists — neither notebook ever ran on a real diffusion LLM. The project actually targets **LLaDA** and **MDLM/masked-diffusion** models. |
| 2 | Assumes Mixture-of-Experts router; Week8_3 **raises** if no `.gate`/`.router` found, and does `raw_router_logits.view(canvas_size, num_experts)` | both, esp. Week8_3 | The real targets **LLaDA-8B and Dream-7B are DENSE** (no router). Week8_3's extractor would crash on them. Router capture must be optional. |
| 3 | Manual loop initializes the canvas with `torch.randint` random token IDs and re-decides **all** positions every step (`current_input_ids[0, prompt_len:] = top_token_ids_t`) | Week8_3 cell 1 | Two precise faults: (a) masked diffusion must init the unknown region to the **`[MASK]` id**, not random tokens (random init belongs to uniform/D3PM diffusion); (b) committed tokens must be **frozen**, unmasking only a subset per step. The bug is the random init + no freezing — *not* the use of argmax (greedy selection is legitimate). Self-declared broken in the notebook. |
| 4 | The stated *reason* for that bug — "diffusion denoises continuous latent embeddings, not discrete IDs" | Week8_3 markdown | **Factually wrong** for the dominant open diffusion LLMs. LLaDA, Dream, MDLM, SEDD are **discrete masked-token** diffusion over token IDs with a `[MASK]` state. (Continuous-latent diffusion — Diffusion-LM, Plaid — is a different, minority family.) The loop is broken for *scheduling* reasons, not because of continuous latents. |
| 5 | `logits[0, -canvas_size:, :]` + "one forward pass == one step" + `pop(0)` | both hook extractors | Model-specific assumptions. They break under KV-caching (per-step logits may cover only new positions), block/semi-AR decoding (canvas grows), or a custom `diffusion_generate()`. |
| 6 | Entropy computed at **every** position **every** step, including frozen and not-yet-reached masked slots | both | Conflates three quantities: (a) genuine uncertainty of a masked-and-being-decided token — the signal you want; (b) a frozen token re-encoded (~0 entropy, exactly 0 under SUBS-parameterised MDMs); (c) a not-yet-reached masked slot, which is confounded in **both** directions — often artificially *low* (mass piling on **`<eos>`/`<pad>`**; note well-trained MDMs suppress the `<mask>` logit, so mask itself isn't the attractor) or artificially *high* (missing local context). Fix: track mask state and restrict/annotate entropy to still-masked positions; optionally exclude `<eos>/<pad>` from the entropy support. |

## 3. The correct, general extraction

Isolate everything model-specific in a small `DiffusionModelSpec` adapter, then use
a framework-owned denoising loop so step index, mask state, and the predictive
distribution are all exact:

- **Backend A — `run_controlled_denoising`** (recommended): reimplements the
  masked-diffusion sampler (a faithful generalisation of LLaDA's official loop),
  computing entropy on the model's distribution at masked positions **before**
  they are committed, and recording `is_masked` / `committed_at_step` per token.
- **Backend B — `capture_via_hooks`** (fallback): hardened version of the
  notebooks' passive-hook method for when you can only call the model's native
  `generate()`/`diffusion_generate()`. Router optional; canvas width and step
  count derived from what's captured; robust prefill filter instead of `pop(0)`.

The heatmap is unchanged in spirit but gains a `mask_aware=True` option that blanks
cells after a token is committed, so the colour reflects real
uncertainty-while-undecided.

Every user-editable point is marked `>>> CUSTOMIZE` in the code, and presets are
provided for LLaDA (`spec_for_llada`) and Dream (`spec_for_dream`).

## 4. Reading the plots (and the LLaDA-8B run)

The representation is correct; here is what each plot means and why an actual
LLaDA-8B run looks the way it does.

- **Denoising heatmap** — token position (rows) × step (cols) × entropy. The
  lower-left triangle (still-masked positions) is high-entropy; the upper-right
  (committed) is blank under `mask_aware=True`. The diagonal is the
  confidence-ordered **unmask wavefront**: the order in which the model becomes
  confident is the real signal here.
- **Convergence curve** — mean entropy over *still-undecided* tokens per step.
  For LLaDA this stays **high (~4.7 nats) for most of the run and collapses only
  in the last ~10–15 steps**. That is genuine and informative: LLaDA commits by
  *confidence rank*, always removing the least-uncertain masked token, so the pool
  of survivors stays uncertain until context is nearly complete.
- **"Aha" histogram** and **freeze curve** — see the three observations below.

### Your three observations — explained

1. **"Early and later tokens sit at constant low entropy, never high."**
   - *Later positions* (~last 5): the response is shorter than `gen_length`, so the
     trailing slots are **padding**, which the model confidently fills with
     `<eos>/<pad>` → low entropy. That is a real artifact, now removed by excluding
     `<eos>/<pad>/<mask>` from the entropy (pass `tokenizer=` to `spec_for_llada`,
     which sets `entropy_exclude_ids`).
   - *Earliest positions* (0–a few): they are the most predictable (sentence
     openings), so they are committed in the first step or two and only ever show
     one high-entropy cell before being blanked. Not a bug.

2. **"The aha histogram is flat — entropy falls at a constant rate."**
   This is **correct behaviour of the default fixed schedule, not a bug.** LLaDA's
   reference sampler commits exactly `gen_length/steps` tokens per step (here 1
   token/step). Since each token's "decision step" is its commit step, the
   histogram is *uniform by construction* — it re-plots the schedule, not the model.

3. **"The freeze curve rises linearly."** Same root cause: a constant number of
   tokens lock in per step, so the cumulative fraction is a straight diagonal. It
   is a property of the **sampler's schedule**, not the model's confidence.

### The fix for observations 2 & 3

Set `schedule='confidence_threshold'` on the spec (now the default in the example).
Instead of committing a fixed count per step, the loop commits **every masked
position whose top-1 confidence exceeds a threshold**, so the number locked per
step is model-driven. The freeze curve becomes a real S-curve and the aha histogram
concentrates on the steps where the model actually gains confidence. Use
`schedule='fixed'` to reproduce the exact LLaDA sampler (and the linear curves).

The heatmap and convergence curve were already correct; observations 2 & 3 were
the two plots that, under a fixed schedule, only re-describe the schedule.
