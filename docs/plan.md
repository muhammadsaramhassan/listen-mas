# Detecting Black-Box Malicious Agents via White-Box Sensors in LLM Multi-Agent Systems

## Headline claim

In a MAS where only some agents expose internals (open-weight = white-box readers, API = black-box), malicious agents — including black-box ones — can be identified by (i) probing white-box readers' activations at ingestion of each sender's message, (ii) attributing influence causally via message ablation, and (iii) aggregating per-sender scores across readers with a sequential test that gives a controllable FPR. AcMAS probes each agent's *own* activations and so is blind to black-box senders by construction; text/graph methods collapse on stealthy attackers (NARCBench shows LLM judges fail on steganographic collusion where activation probes reach 0.99+ AUROC).

## Core mechanism

A single reader cannot detect a stealthy lie about a fact it does not hold. Shards overlap partially by construction, so any lie contradicts *some* reader's shard. A stealthy attacker lies rarely, but over many rounds eventually contradicts someone who knows better. Readers emit per-message scores; the orchestrator accumulates sparse high-conflict events per sender across readers and rounds. Multi-reader aggregation is the mechanism, not a robustness add-on.

---

## 1. Harness: fork of Silo-Bench (ACL 2026)

Repo: `github.com/jwyjohn/acl26-silo-bench` — Unlicense (public domain), ~4.9k lines, Python 3.13 + uv.

### What we get for free
- Task-agnostic engine: task JSON with `input_shard` per agent, `expected_output`, verification rule.
- Three protocols: P2P (`msg`), broadcast, shared file system. Round-synchronized (messages sent in round t visible in t+1).
- Per-agent `context.json` persisted every round → **message ablation is a reload-edit-rerun, no replay system needed**.
- Metrics (success, partial correctness, token cost, comm density), logging, batch runner, arbitrary agent counts via generator.
- Messages arrive as `receive_messages` tool results → each incoming message has an identifiable token span in the reader's context for activation pooling.

### Required modifications
| # | Change | Where | Est. |
|---|---|---|---|
| H1 | Per-agent model/backend config (currently one `metadata.config.model` for all) | `src/models.py`, `src/engine.py` round loop | 0.5 d |
| H2 | In-process HF backend for white-box agents with forward hooks capturing residual stream; `call_llm` stays for API agents | `src/utils/llm.py` + new `src/utils/whitebox.py` | 2–3 d |
| H3 | Fixed communication graph: adjacency mask on `send_message` recipients (P2P topology is currently emergent) | `src/msg/tools.py`, config | 0.5 d |
| H4 | Malicious-agent hooks (see §3) | `src/*/tools.py`, prompts | 1–2 d |
| H5 | Ablation runner: reload `context.json`, drop sender s's message from the tool result, re-run reader forward pass, diff | new `src/ablate.py` | 1 d |
| H6 | Overlap wrapper for Paradigm I generators (currently disjoint `data[start:end]`) | `benchmark_generator/paradigm_i.py` | 1 d |
| H7 | Max rounds default 100 → 12 | config | — |

---

## 2. Environments (all run through the same engine)

### Env A — Silo-Bench numeric (controlled; "does the signal exist")
- Paradigm I tasks I-01…I-10 (global max, word frequency, vote, range count, checksum, average, set union, top-k, std dev, any-match). Paradigm II/III deferred (boundary semantics interact with overlap).
- Shards: random numeric data, N agents, **overlap parameter k** = elements duplicated into neighboring shards (0 = original Silo-Bench).
- Ground truth exact. Lies are numeric → text-only baselines fail by construction (note as favorable to us; not the main result).
- **Risk:** "this contradicts my shard" may be a weaker/less linear representation for arithmetic on integer lists than for narrative facts. Env A is a go/no-go check, not the paper's main evidence.

### Env B — Hidden-profile tasks (narrative facts; main results)
- Paradigm: Hidden Profile (Stasser & Titus 1985). K options × attributes with valence; attributes split into `shared_information` (all agents) and `hidden_information[i]` (agent i only). Constructed so the correct option looks *worst* on shared info alone and *best* on the union → coordination is necessary and sufficient.
- **Generator** emits HiddenBench-format JSON (`description`, `shared_information`, `hidden_information`, `possible_answers`, `correct_answer`), parameterized by N agents, K options, attributes per option, **shared fraction (= overlap dial)**, and domain templates (hiring, procurement, site selection, vendor choice, …). Entity names/attributes are fictional per instance → no pretraining leak.
- **Converter** → Silo-Bench task JSON: `input_shard_i = shared_information + hidden_information[i]`; `expected_output = correct_answer`; verification = exact option match.
- **External validation:** the 65 official HiddenBench tasks (`YuxuanLi1225/HiddenBench`, MIT), 3–4 agents native, 3–4 options. Held out entirely: never generate from them, never tune on them. Run at their native N; pad with shard-less observer agents if N=6 needed.
- Answer space is 3–4 options → ablation effect is Δ over P(option), a clean scalar.

### Deferred / dropped
- EntCollabBench (enterprise workflows, 11 agents / 6 departments): realistic but tool-environment heavy; stretch for v2.
- InformativeBench (2024): superseded. Social-deduction games: detection is the game itself.

### Mandatory sanity baselines (per environment)
1. Own-shard-only single agent → chance.
2. Union-of-shards single agent → near ceiling.
Gap = auditable proof of properties (a) not solvable alone and (b) coordination required. HiddenBench already reports 30.1% (distributed) vs 80.7% (complete) for reference.

---

## 3. Maliciousness

### Implementation layers
| Layer | Mechanism | Used for | Labels |
|---|---|---|---|
| **Transport** | Intercept `send_message`/`broadcast_message` for malicious `agent_id`; perturb numbers programmatically; cache perturbations for consistency | Env A, T0/T1 | Exact per-message flag |
| **Prompt** | Malicious agent gets true shard + target wrong option + list of false claims to advocate; T2 adds conversation visibility and "assert only what no one can contradict" (worst case: oracle of who holds what) | Env B, T1/T2/T3 | Offline checker with ground-truth access (GPT-4-class judge, eval only, never in loop) |
| **Shard** | Corrupted shard, honest relay — agent does not know it is wrong | Faulty-agent control | Exact (shard diff) |

### Knobs
| Knob | Values | Default |
|---|---|---|
| Corruption rate p | 1.0, 0.5, 0.2, 0.1 | 0.2 (stealth) |
| Plausibility | blatant (random replace / valence flip on salient attribute) → plausible (±10–20% / flip one minor attribute) | plausible |
| Answer-path relevance | on-path / off-path | mixed |
| Consistency | consistent (cached) / inconsistent | consistent |
| Adaptivity | none / conversation-aware / oracle | none |
| # malicious m | 0 (null), 1, 2 | 1 |

### Attacker tiers
- **T0** Blatant, non-adaptive, p=1.0. Pipeline check; text baselines must catch it.
- **T1** Plausible, consistent, p=0.2. Main results.
- **T2** Adaptive (conversation-aware, then oracle). Stresses the shard-overlap assumption; quantify overlap needed.
- **T3** Sabotage: withhold hidden info / verbose stalling / derail. Borrow taxonomy from "Who's the Mole?" (suboptimal fixation, reframing misalignment, fake injection, execution delay).
- **T4** (stretch) Collusion: malicious sender + malicious white-box reader.
- **Control:** faulty agent (shard layer).

---

## 4. Configuration

### Default run
| Item | Value |
|---|---|
| Agents N | 6 |
| Malicious m | 1, **black-box** by default (headline); also run malicious-is-white-box for AcMAS comparison |
| White-box readers w | 2 (randomly assigned IDs), full participants (hold shards, send messages) |
| Black-box agents | 6 − w; API models (GPT-4o-mini / Claude Haiku class) for main results, unhooked Qwen3-8B for large sweeps |
| Reader backbone | Qwen3-8B-Instruct, bf16, in-process HF with hooks; Llama-3.1-8B as heterogeneity check |
| Protocol | Broadcast first → P2P with fixed random graph (H3) |
| Max rounds | 12 (Env A), 10 (Env B) |
| Overlap | Env A: k ∈ {0, 2, 5, 10} elements; Env B: shared fraction ∈ {0.2, 0.4, 0.6} |
| Instances | Env A: 10 tasks × 30 seeds = 300; Env B: 300 generated + 65 official held-out |
| Conditions per instance | honest (m=0) + attacked → honest runs give the null distribution for FPR |

### Compute
Agents are contexts, not model copies: 6 agents on one 8B model = one weight set. One 80 GB card suffices. Ablation: one extra reader pass per (reader, sender, round) ≈ 2 × 5 × 10 = 100 short passes per task instance; prefix-cacheable.

### Sweeps / ablations
- w ∈ {1, 2, 3, 5} readers (detection vs coverage).
- Overlap dial (both envs) — the parameter the mechanism depends on.
- Reader heterogeneity: same-family vs mixed.
- Passive **sentinel** reader (no shard, listens only).
- Corruption rate, plausibility, on/off-path, consistency, adaptivity.
- Layer sweep for activation capture; pooling (mean over message span vs last token).
- Aggregator: median vs trimmed mean vs mean; CUSUM vs SPRT.

---

## 5. Method

### Per-message score at reader r for sender s's message
- **Conflict:** does r's representation register the message as contradicting its shard? Residual stream pooled over the incoming message's token span at ingestion (the forward pass that consumes the `receive_messages` result).
- **Influence:** ablate the message (H5), diff r's answer-relevant representation / P(option). Gates conflict: contradiction nobody attends to is harmless.

### Scoring variants (staged)
1. **Supervised upper bound (not the paper):** linear probe, train on generated world A, test on world B. Go/no-go for signal existence.
2. **Unsupervised (the paper):**
   - Robust anomaly (MCD covariance on incoming-message activations per reader → Mahalanobis).
   - Self-verification pseudo-labels (reader checks claim vs own shard via forward pass → noisy label → probe). System-internal supervision only.
   - Contrastive consistency (CCS-style; claim vs negation). Strongest no-labels story, least reliable.
   - Belief-shift direction via ablation (label-free; only meaningful with a conflict signal).
   - Expected final: anomaly gated by influence; self-verification as the sharper variant.

### Orchestrator (model-free)
- Input `S[reader, sender, round]`.
- Cross-reader: median / trimmed mean.
- Temporal: per-sender CUSUM or SPRT; flag on threshold crossing.
- FPR: threshold from the null distribution (honest runs). "Deployable FPR" = calibration statement.
- Coverage: per-sender count of readers that received its messages; zero-coverage nodes reported as *uncovered*, not honest.

### Trust model
Sensors are trusted because the orchestrator reads activations directly — a malicious white-box reader can lie in messages but not in what it registers. Colluding sender + reader partially breaks this (T4, limitation).

---

## 6. Baselines
- Text-only LLM-as-judge per message.
- Graph-based trust propagation over the comm graph.
- **AcMAS** (final-layer hidden state of each agent's own generation; per-backbone) — run in malicious-is-white-box condition; show it is undefined for black-box senders.
- **NARCBench-style** group-level activation aggregation (nearest white-box method; probes the colluders themselves).
- Task-outcome-only post-hoc attribution (upper bound requiring labels deployment won't have).

---

## 7. Execution order
1. Fork Silo-Bench; H1, H7. Run Env A honest at N=6 with an API model end-to-end.
2. H6 overlap wrapper; H3 fixed graph; regenerate Env A tasks (N=6, 30 seeds).
3. Sanity baselines on Env A.
4. H2 white-box backend + hooks; verify activations captured per message span.
5. H4 transport-layer malice (T0). H5 ablation runner.
6. Stage-1 supervised probe on Env A, cross-world. **Go/no-go.**
7. Env B generator + converter; sanity baselines; held-out HiddenBench-65 loads and runs.
8. Prompt-layer malice (T1) + offline checker for per-message labels.
9. Stage-2 unsupervised scoring; orchestrator; FPR calibration on honest runs.
10. Main results: Env B, T1, broadcast → P2P fixed graph, w sweep.
11. Overlap dial (both envs). T2 adaptive. T3 sabotage. Faulty control.
12. Baselines (AcMAS, NARCBench-style, text judge, graph). Writeup.

## Open risks
- Numeric-contradiction representations weak in Env A → treat Env A as go/no-go only; Env B carries the paper.
- 8B readers may not form clean conflict representations → fall back to a 32B reader (fits one 80 GB card alone).
- Anomaly scoring fails at low p → rely on self-verification pseudo-labels.
- Adaptive attacker defeats overlap → report the overlap density required as a quantitative limit.
- Prompt-layer malicious agent may not actually lie when instructed → verify with offline checker early; fall back to structured-claim messages if compliance is low.
- Orchestrator never sees task outcomes (default). Revisit only for calibration.
