# listen-mas

Environment for **listener-side detection of malicious agents** in LLM multi-agent systems
where some agents are open-weight (white-box readers, activations readable) and others are
black-box (API, or open weights used as a proxy). Detection probes the *readers'* activations
at message ingestion; senders are never touched, so black-box liars are in scope.

Companion documents: `docs/plan.md` (environment), `docs/protocol.md` (pre-registered tests,
orchestrator), `docs/paper_plan.md`, `docs/context_and_audit_prompt.md`.

## What is in here

```
listen/
  tasks/         hidden_profile.py  procedural hidden-profile generator (5 domains, fictional names,
                                    duplication dial d, hidden-profile property enforced)
                 convert.py         Silo-Bench numeric tasks -> Task; official HiddenBench -> Task
  backends/      hf_backend.py      HF transformers: white-box (hooks on every layer: residual, attn-out,
                                    mlp-out; mean + last-token pooling over each [MSG] span) and
                                    hf_blackbox proxy (same weights, shared in memory, no hooks)
                 api_backend.py     OpenAI-compatible (OpenAI, OpenRouter, local vLLM server)
                 mock_backend.py    scripted agents for CPU tests
  engine/        runner.py          round-synchronised MAS loop; capture at ingestion; persistence
                 protocol.py        <message to=...>, <answer>, [MSG id= from=] delivery template
                 comm_graph.py      complete / ring / random_k / explicit; ejection
  malice/        roles.py           prompt-layer attacker tiers T0/T1/T2/T2o, faulty control, T3 sabotage
                 transport.py       numeric corruption of outgoing messages (Silo env), per-message labels
                 labeler.py         heuristic + LLM-judge message labelling (eval only)
  capture/       store.py           activations/round_t/reader/msg.npz + index.jsonl
  orchestrator/  core.py            scorers (probe / Mahalanobis), calibration on honest runs, per-reader
                                    z-score -> median across readers -> CUSUM -> quorum ejection; vote baseline
  ablate.py                         same-round single-reader message ablation (delta P(option))
  probe_tests/   gen_items.py       pre-registered signal-test items S1,S3,S4,S7,S8 (valence-balanced)
                 run_probe.py       capture -> layer x component x pooling sweep -> S1-S9 report + best probe
  experiment.py                     YAML config -> tasks, agents, roles, graph, runner, orchestrator
scripts/         probe_tests.py run_experiment.py gen_tasks.py sanity_baselines.py calibrate.py evaluate.py ablate.py
configs/         mock_smoke.yaml pilot_local_hf.yaml main_broadcast_T1.yaml graph_random2_T1.yaml
                 silo_numeric_T1.yaml hiddenbench_heldout.yaml
data/            hiddenbench_benchmark.json (65 official tasks, held-out validation only)
tests/           CPU test suite (scripted agents + tiny random model)
```

## Setup

### Any machine (M5 dev, CPU only)
```bash
cd listen-mas
python -m venv .venv && source .venv/bin/activate          # or: uv venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest
python -m pytest tests -q -m "not slow"                     # ~1 s: generator, protocol, malice, mock end-to-end
python -m pytest tests -q -m slow                           # ~30 s: tiny random model through capture + probe
python scripts/run_experiment.py --config configs/mock_smoke.yaml
```

### GPU box (A100/H100, CUDA 12.x)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt pytest
export HF_HOME=/path/with/space/hf_cache                    # models cache here (Qwen3-8B ~16 GB)
huggingface-cli download Qwen/Qwen3-8B                      # optional pre-download
python -m pytest tests -q -m "not slow"
```
Optional, for throughput on black-box proxies: `pip install vllm` and run
`vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.45` on a second GPU (or same 80 GB card
alongside the HF copy), then use `{type: api, model: Qwen/Qwen3-8B, base_url: http://localhost:8000/v1, api_key_env: NONE}`
for `blackbox`, and add `gen_backend: {same api dict}` inside the `whitebox` entry to delegate the readers'
text generation too (capture still happens on the HF copy).

### Local box (2x RTX 6000 + V100)
* RTX 6000 (Turing, 24 GB) / RTX 6000 Ada (48 GB): use `dtype: float16` on Turing (no bf16), bf16 fine on Ada.
  Qwen3-8B fp16 = ~16 GB weights + activations; fits on 24 GB with short contexts (rounds <= 8, `max_new_tokens: 300`).
* V100 (16/32 GB): **`dtype: float16` is required** (no bf16). 16 GB card cannot hold 8B; use the 32 GB one,
  or run a 4B reader (`Qwen/Qwen3-4B`) for plumbing tests only. Pass `--dtype float16` to scripts.
* `device_map: auto` spreads one model over multiple GPUs if needed (`CUDA_VISIBLE_DEVICES=0,1`).
* This box is for pilots and Step 0 at reduced scale (`--n-worlds 40`); main runs go to the H100.

### Migrating between machines
The repo is self-contained; copy it with `rsync -a --exclude .venv --exclude runs listen-mas/ host:~/listen-mas/`.
Results live in `runs/` and `results/` (rsync those back). Model weights are in `$HF_HOME`; set the same
`HF_HOME` on each machine or let each download once. `HF_HUB_OFFLINE=1` after download for air-gapped nodes.

## Run order

### Step 0 — signal tests (must pass before anything else; see docs/protocol.md B)
```bash
python scripts/probe_tests.py --model Qwen/Qwen3-8B --no-thinking --n-worlds 80 --per-world 8 \
       --out results/probe_qwen3_8b
```
* Generates ~5k items (S1 1280, S3 640, S4 640, S7 1920, S8 640) from world seeds 5000+, captures every layer
  (residual, attention-out, MLP-out) x (mean, last) over the `[MSG]` span, sweeps layer x component x pooling on the
  selection half of the worlds, reports on the other half with bootstrap CIs, runs S2/S5/S6/S9, and writes
  `results/probe_qwen3_8b/report/report.md` + `best_probe.npz` (used by the orchestrator).
* H100: roughly 20-40 min. Storage ~1.8 MB/item -> ~9 GB; use `--components resid` to cut to a third for a first pass.
* Resumable: rerun the same command and cached `<test>.npz` are reused. `--skip-s9` avoids the generation loop.
* If S1 is MARGINAL/FAIL: `--model Qwen/Qwen3-32B` (80 GB card, bf16) per the decision rule.

### Step 1 — tasks and sanity baselines
```bash
python scripts/gen_tasks.py --out tasks/train --n 200 --seed0 0
python scripts/gen_tasks.py --out tasks/eval  --n 300 --seed0 1000
python scripts/sanity_baselines.py --tasks-dir tasks/eval --model Qwen/Qwen3-8B --no-thinking --limit 40
```
Own-shard accuracy must be ~chance (0.25 for 4 options) and union accuracy near ceiling; otherwise adjust
`--shared-fraction` / `--duplication` (see note below) before running MAS experiments.

### Step 2 — MAS runs (honest + attacked)
```bash
python scripts/run_experiment.py --config configs/pilot_local_hf.yaml                    # T0 pilot, 20 tasks
python scripts/run_experiment.py --config configs/main_broadcast_T1.yaml --limit 50       # T1 main
python scripts/run_experiment.py --config configs/main_broadcast_T1.yaml --set malice.tier=T2 run_name=main_T2
```
Each run directory contains `task.json`, `roles.json` (who is white-box, who is malicious, the false claims),
`rounds/round_t/agent/{context,output,parsed}`, `messages.jsonl`, `labels.jsonl` (per-message false-claim labels for
the malicious sender), `activations/`, `summary.json`.

### Step 3 — calibrate, evaluate, then run with the online orchestrator
```bash
python scripts/calibrate.py --honest-glob "runs/main_broadcast_T1/honest/*" --scorer probe \
       --scorer-path results/probe_qwen3_8b/report/best_probe.npz --n-rounds 10 --target-fpr 0.05 \
       --out runs/main_broadcast_T1/calibration.json
python scripts/evaluate.py --attacked-glob "runs/main_broadcast_T1/attacked/*" --honest-glob "runs/main_broadcast_T1/honest/*" \
       --calibration runs/main_broadcast_T1/calibration.json --scorer probe \
       --scorer-path results/probe_qwen3_8b/report/best_probe.npz --out results/eval_T1.json
# online ejection (uses the calibration above):
python scripts/run_experiment.py --config configs/main_broadcast_T1.yaml --set orchestrator.enabled=true \
       run_name=main_T1_online conditions=[attacked]
```
`evaluate.py` reports sender-level AUROC, detection rate and time-to-detection at the calibrated threshold,
honest false-flag rates, coverage (fraction of malicious senders that reached quorum), and the Among-Us vote baseline.

### Ablation and other environments
```bash
python scripts/ablate.py --runs-glob "runs/main_broadcast_T1/attacked/*" --model Qwen/Qwen3-8B --no-thinking
python scripts/run_experiment.py --config configs/silo_numeric_T1.yaml          # Env A: numeric, transport corruption
python scripts/run_experiment.py --config configs/hiddenbench_heldout.yaml      # external held-out validation
python scripts/run_experiment.py --config configs/graph_random2_T1.yaml         # sparse comm graph
```

## Notes and design facts

* **Ingestion capture.** A reader's activations over each `[MSG id=... from=...]` block are taken from the
  prefill forward pass of the turn in which that block first appears. Six arrays per message:
  `resid_{mean,last}` `[L+1, D]`, `attn_{mean,last}` `[L, D]`, `mlp_{mean,last}` `[L, D]`, float16.
  For Qwen3-8B (L=36, D=4096) that is ~1.2 MB/message; a 6-agent 10-round run with 2 readers stores ~100 messages.
  Restrict with `components:` / `poolings:` / `layers:` in the backend config for large sweeps.
* **Duplication leaks information.** With N=6, `d=2` keeps an agent's own-shard heuristic accuracy at chance
  (~0.29 vs 0.25); `d=3` raises it to ~0.5 because each agent then holds half the hidden facts. Use
  `--individual-misleading` (rejects tasks where any single agent's own view already identifies the answer) when
  sweeping d, and always re-run `sanity_baselines.py` per d.
* **Qwen3 thinking.** Pass `chat_template_kwargs: {enable_thinking: false}` (or `--no-thinking`), otherwise
  outputs start with `<think>` blocks and consume the token budget.
* **Seeds.** Task worlds: train 0-199, eval 1000+, probe tests 5000+. Never mix; splits are by world id.
* **Black-box agents.** Three interchangeable options in configs: `hf_blackbox` (same weights as the reader, no
  hooks), `api` with a `base_url` (local vLLM), or `api` against a commercial provider. Mixed fleets are fine
  (the config takes one `blackbox` entry; edit `experiment.build_agents` for per-agent heterogeneity).
* **Speed.** HF `generate` runs ~40 tok/s for 8B; a 6-agent x 8-round run is ~5-7 min. Use vLLM for black-box
  proxies and `gen_backend` delegation for readers to cut this to ~1 min/run.
* **Labels.** Transport-layer corruption gives exact per-message labels. Prompt-layer malice is labelled by
  `labeler.heuristic_label` (matches the assigned false-claim renderings); add an LLM judge via `labeler.llm_label`
  for the paper's evaluation set. Sender-level labels (`roles.json`) are exact either way.
