"""CPU-only tests. Run: python -m pytest tests -q   (the tiny-model tests download the Qwen2.5-0.5B tokenizer once)."""
import glob, json, os, shutil, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listen.tasks.hidden_profile import generate_task, to_hiddenbench, from_hiddenbench
from listen.tasks.convert import hiddenbench_official
from listen.engine.protocol import parse_turn, normalise_answer
from listen.engine.comm_graph import CommGraph
from listen.malice.roles import make_malicious, make_faulty
from listen.malice.transport import NumericCorruptor
from listen.engine.protocol import Message
from listen.experiment import load_config, run_experiment
from listen.orchestrator.core import DimScorer, calibrate, score_run, sender_trajectories

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_generator_property():
    for s in range(30):
        t = generate_task(s, duplication=2)
        tot = t.meta["totals"]; st = t.meta["shared_totals"]; c = t.correct_answer
        assert tot[c] == max(tot.values()) and sum(v == tot[c] for v in tot.values()) == 1
        assert c in sorted(st, key=st.get)[:2] and max(st, key=st.get) != c
        for f in t.facts.values():
            assert (len(f.holders) == t.n_agents) if f.shared else (len(f.holders) == 2)
        hb = to_hiddenbench(t); back = from_hiddenbench(hb)
        assert back.n_agents == t.n_agents and back.correct_answer == t.correct_answer


def test_hiddenbench_import():
    ts = hiddenbench_official(os.path.join(ROOT, "data", "hiddenbench_benchmark.json"))
    assert len(ts) == 65 and all(t.n_agents in (3, 4) for t in ts)


def test_protocol_parse():
    p = parse_turn('hello <message to="all">A is good</message> <message to=agent_2>psst</message> <answer>B</answer>', "agent_0", 1)
    assert len(p.messages) == 2 and p.messages[1].to == "agent_2" and p.answer == "B"
    assert normalise_answer("Option C", ["A", "B", "C", "D"]) == "C"
    assert normalise_answer("candidate b", ["Candidate A", "Candidate B"]) == "Candidate B"


def test_graph():
    ids = [f"agent_{i}" for i in range(6)]
    g = CommGraph.random_k(ids, 2, seed=1)
    assert all(len(g.neighbours(a)) >= 2 for a in ids)
    g.eject("agent_0")
    assert "agent_0" not in g.active() and g.receivers("agent_0", "all") == []
    assert "agent_0" not in g.receivers("agent_1", "all")


def test_malice_roles_and_transport():
    t = generate_task(7, duplication=2)
    r = make_malicious(t, "agent_1", tier="T1", rate=0.2, n_claims=3)
    assert r.claims and all(c.holders for c in r.claims)      # contradictable
    assert r.target != t.correct_answer and "SECRET" in r.role_prompt
    f = make_faulty(t, "agent_1", n_corrupt=2)
    assert f.shard_text != t.agent("agent_1").shard_text
    c = NumericCorruptor(rate=1.0, mode="plausible", seed=0)
    m = c(Message("x", "agent_0", "all", "my max is 827 and min is -949", 1))
    assert m.flags["corrupted"] and m.content != "my max is 827 and min is -949"
    m2 = c(Message("y", "agent_0", "all", "again 827", 1))
    assert dict(m.flags["changes"])["827"] == dict(m2.flags["changes"])["827"]   # consistent


def test_mock_end_to_end_and_orchestrator(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "mock_smoke.yaml"))
    cfg["out_dir"] = str(tmp_path); cfg["tasks"]["generate"]["n"] = 6
    res = run_experiment(cfg)
    assert len(res) == 12
    honest = sorted(glob.glob(str(tmp_path / "mock_smoke/honest/*")))
    attacked = sorted(glob.glob(str(tmp_path / "mock_smoke/attacked/*")))
    assert all(os.path.exists(d + "/activations/index.jsonl") for d in honest + attacked)
    assert all(os.path.exists(d + "/labels.jsonl") for d in attacked)
    sc = DimScorer("resid", "mean", 2, 0)
    cal = calibrate(honest, sc, n_rounds=6, target_fpr=0.05)
    hits = 0
    for d in attacked:
        mal = json.load(open(d + "/roles.json"))["malicious"]
        tr = sender_trajectories(score_run(d, sc), cal, cal.drift, 6)
        hits += tr[mal]["max"] > cal.threshold
    assert hits >= 4   # planted signal must be recovered in most runs


@pytest.mark.slow
def test_tiny_model_capture_and_probe(tmp_path):
    from listen.backends import build_backend, Span
    from listen.probe_tests.gen_items import gen_items
    from listen.probe_tests.run_probe import capture_items, run_tests
    wb = build_backend({"type": "whitebox", "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float32",
                        "device_map": None, "tiny_random": True})
    blk = "[MSG id=abc from=agent_1] Candidate B is unreliable. [/MSG]"
    caps = wb.capture_only("sys", [{"role": "user", "content": "New messages:\n" + blk + "\nReply."}], [Span("abc", blk)])
    assert caps and caps[0].arrays["resid_mean"].shape[0] == wb.n_layers + 1
    items = gen_items(list(range(4)), per_world=2, tests=("S1",))
    paths = capture_items(items, wb, str(tmp_path))
    rep = run_tests(paths, str(tmp_path / "out"), backend=None)
    assert "S1" in rep["tests"] and os.path.exists(tmp_path / "out" / "best_probe.npz")
