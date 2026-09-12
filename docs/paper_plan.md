# Paper Plan — Listener-Side Detection of Black-Box Malicious Agents

**Target:** ICML 2027 (abstract Jan 16, full paper Jan 22, 2027).
ICLR 2027 (Sept 25, 2026) is infeasible for this scope. Optionally register an ICLR abstract as a fallback only if the go/no-go (§6, step 0) passes within days.

**Working title:** *Reading the Room: Detecting Black-Box Malicious Agents from the Activations of Their Listeners*
**Method name (placeholder):** LISTEN — Listener-side Inference of Sender Trust via Ensembled Neural probes.

---

## 1. One-paragraph pitch

Deployed multi-agent systems mix open-weight and API models. Existing activation-based defenses (AcMAS, NARCBench-style probes) read the *sender's* internals and are therefore blind to any API agent. We show that a malicious sender leaves a trace in the *listeners*: when a lie contradicts something a white-box reader knows, the reader's residual stream registers the conflict even when its text does not. Aggregating these traces across readers and rounds with a sequential test yields detection of black-box attackers at a controlled false-positive rate. We give a simple detectability theory (detection depends on shard-overlap density, corruption rate, number of readers, rounds), show it predicts empirical curves, and release a modified hidden-profile MAS benchmark with controllable overlap and malice.

---

## 2. Contributions (exactly these four)

**C1. Problem: partially observable MAS security.** Formalize detection in a MAS with a white-box subset W and black-box subset B, communication graph G, and per-agent information shards with overlap. Define *coverage* of a sender as the set of readers in W that receive its messages and hold overlapping facts. Prove (trivially) that speaker-side methods have zero coverage of B.

**C2. Method: LISTEN.** Per-message score at reader r for sender s = conflict(r, m) × influence(r, m): conflict from a linear readout of r's residual stream over m's token span at ingestion (unsupervised variant: robust Mahalanobis; system-supervised variant: self-verification pseudo-labels); influence from same-round message ablation on r's answer distribution. Cross-reader robust aggregation (median), per-sender sequential test (CUSUM/SPRT) with threshold calibrated on honest runs → controlled FPR. Coverage reported per sender; uncovered ≠ honest.

**C3. Theory: detectability.** Per-round hit probability h ≈ ρ · p · (1 − (1 − q)^w) where ρ = probability a lie lands on a fact some reader holds (overlap density), p = corruption rate, q = per-reader probe recall, w = readers receiving s. Expected time-to-detection under SPRT ≈ f(h, target FPR, probe FPR). Two predictions: (i) detection vs. overlap is monotone and saturates; (ii) an oracle-adaptive attacker (lies only on uncovered facts) has h = 0 — a stated identifiability limit.

**C4. Benchmark: HP-MAS.** Built on the Silo-Bench engine. Env B (main): procedurally generated hidden-profile decision tasks with **hidden facts duplicated across d ≥ 2 agents** (d = overlap dial) and a malicious role at three tiers. Env A (controlled): Silo-Bench Paradigm I numeric tasks with overlap. Held-out external validation on the 65 official HiddenBench tasks. Released with per-message ground-truth labels.

---

## 3. Claims → evidence map

| Claim | Figure/Table | Baselines |
|---|---|---|
| Black-box attackers are detected at 1–5% FPR (T1, p=0.2) | Fig 1: ROC + time-to-detection, Env B | Text cross-check (fair), LLM-judge (message only), graph trust, AcMAS (undefined for B — shown as N/A) |
| Activations register conflicts that text does not | **Fig 2 (the paper):** per-message conflict-probe score vs. reader's verbalized disagreement; fraction of true lies caught by probe but not verbalized | Text cross-check with identical information |
| Detection follows the theory | Fig 3: detection vs. overlap d (Env A: k; Env B: d) with theory curve overlaid | — |
| Degrades gracefully with coverage | Fig 4: detection vs. #readers w ∈ {1,2,3,5}; broadcast vs. fixed sparse graph | — |
| Stealth hurts text methods more than LISTEN | Fig 5: AUROC vs. corruption rate p and plausibility, all methods | all |
| Identifiability limit | Fig 6: non-adaptive / conversation-aware / oracle-adaptive attacker | — |
| Speaker-side comparison | Table: malicious-is-white-box condition, LISTEN vs. AcMAS-style speaker probe | AcMAS |
| Layer / pooling / aggregator choices | Appendix | — |

**Report explicitly:** honest-agent false-claim base rate per environment (from null runs). This is what the sequential test must overcome.

---

## 4. Fair baselines (non-negotiable)

1. **Text cross-check (primary):** each reader, given its own shard + incoming message, is prompted "Does this contradict what you know? yes/no + confidence." Same aggregation and sequential test downstream. This isolates *activations vs. verbalization* holding information constant.
2. **Message-only LLM judge** (weak; expected to fail on T1).
3. **Graph-based trust propagation** over the comm graph using text-derived agreement.
4. **Speaker-side activation probe (AcMAS-style)** — only in the malicious-is-white-box condition; reported as N/A for black-box attackers.
5. **Outcome-based post-hoc attribution** — upper bound using labels deployment will not have.

---

## 5. Framing decisions (pre-empting reviews)

- **"Incremental over NARCBench/AcMAS":** the object probed is different (listener conflict vs. speaker deception); the theory section and the coverage formalism are the scientific content, not the coverage argument alone.
- **"Hidden profile makes the attacker uncontradictable":** yes — hence hidden facts are duplicated across d agents; the benchmark is a *modified* hidden profile and says so. Alternative attack surface: lies about shared facts.
- **"Text baseline is a strawman":** the primary baseline has identical information to the readers. If it matches LISTEN, report it; the paper then rests on cost and on stealth robustness, and says so.
- **"Faulty vs. malicious":** out of scope by construction; the method detects *harmful senders*. Dropped from main experiments.
- **"Ablation isn't counterfactual":** influence is a same-round, single-reader quantity; never called task-level attribution.
- **"Readers might be malicious too":** sensors are trusted because the orchestrator reads activations directly; collusion (T4) is a stated limitation with a small experiment if time permits.
- **"Evaluation awareness":** cite the Sept 2026 result that monitored agents change behavior; note our attackers are not told about monitoring; discuss.

---

## 6. Execution schedule (≈18 weeks to Jan 22)

**Step 0 — Go/no-go (this week, no harness).**
Generate ~500 (shard, consistent message, contradicting message) triples in the hidden-profile domain. Qwen3-8B, single forward pass, pool residual over message span, layer sweep, linear probe, train/test on disjoint fictional worlds. **Pass: AUROC ≥ 0.80 at the best layer. Marginal: 0.70–0.80 → try Qwen3-32B before deciding. Fail: < 0.70 on 32B → activation-based paper does not exist; pivot to self-verification-only or abandon.**
Also measure: how often the same model *verbalizes* the contradiction when asked. Early read on Fig 2.

**Weeks 1–3 — Harness.** Fork Silo-Bench; per-agent backends (H1); in-process white-box backend with hooks (H2); fixed comm graph (H3); overlap wrapper (H6); ablation runner (H5). Env A honest runs at N=6.

**Weeks 3–5 — Env B.** Hidden-profile generator with duplication d; converter; sanity baselines (own-shard ≈ chance, union ≈ ceiling); HiddenBench-65 loads and runs held-out.

**Weeks 5–7 — Malice.** Transport-layer (Env A), prompt-layer (Env B) with offline checker labels; T0 and T1. Honest null runs for base rates and FPR calibration.

**Weeks 7–10 — Method.** Supervised probe upper bound; unsupervised variants; influence via ablation; aggregation + SPRT. Fig 1, Fig 2.

**Weeks 10–13 — Sweeps.** Overlap dial (Fig 3), readers (Fig 4), stealth (Fig 5), adaptive (Fig 6). Theory fit.

**Weeks 13–15 — Baselines.** Text cross-check, judge, graph, speaker-probe condition.

**Weeks 15–18 — Writing, ablations appendix, code release, buffer.**

---

## 7. Default configuration

N = 6 agents; m = 1 malicious (black-box by default); w = 2 white-box readers (Qwen3-8B, bf16, in-process, full participants); black-box = API models (main) / unhooked open-weight (sweeps); broadcast → fixed sparse graph; 10 rounds; Env B: 300 generated instances + 65 held-out; Env A: 10 tasks × 30 seeds; every instance run honest and attacked. One 80 GB GPU.

---

## 8. Kill criteria and pivots

| Observation | Decision |
|---|---|
| Step-0 probe AUROC < 0.70 on 32B | Abandon activation framing |
| Probe works but text cross-check matches it everywhere | Reframe as cost + stealth-robustness paper; Fig 2 becomes a null result reported honestly |
| Honest false-claim base rate > attacker corruption rate | Raise p in main results; report stealth limit as a function of base rate |
| Oracle-adaptive attacker at h = 0 but conversation-aware attacker also ≈ 0 | Overlap density required is impractical → report as the paper's negative finding on realistic topologies |
