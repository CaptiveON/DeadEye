"""Exercises the System One (Jev-compatible) client against a local fake server."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from deadeye.envs import make_env
from deadeye.models.base import Message
from deadeye.models.decision_api import DecisionAPIModel
from deadeye.models.registry import load_model
from deadeye.policies import make_policy
from deadeye.runner import run_episode


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        Handler.calls.append((self.path, dict(self.headers), body))
        q = body["questions"]["action"]
        labels = list(q["criteria"])
        # a fake decision model that prefers the last listed option
        probs = {c: 0.1 for c in labels}
        probs[labels[-1]] = 1.0 - 0.1 * (len(labels) - 1)
        resp = {"answers": {"action": {"choice": labels[-1], "probabilities": probs, "confidence": 0.8}},
                "usage": {"input_tokens": 321}, "latency_ms": 12.5}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/systemone"
    srv.shutdown()


def test_state_strips_answer_instruction():
    msgs = [Message("system", "Rules."), Message("user", "Situation.\nLegal actions: a, b\nAnswer with only the action name of your chosen action.")]
    state = DecisionAPIModel.state_text(msgs)
    assert state.startswith("Rules.") and state.endswith("Legal actions: a, b") and "Answer with only" not in state


def test_score_choices_maps_probabilities(server):
    m = DecisionAPIModel("jev", base_url=server, api_key="k")
    res = m.score_choices([Message("system", "s"), Message("user", "u")], ["hit", "stand"])
    assert res.logprobs[1] > res.logprobs[0] and abs(math.exp(res.logprobs[1]) - 0.9) < 1e-9
    assert res.prompt_tokens == 321 and res.extra["confidence"] == 0.8 and res.extra["choice"] == "stand"
    path, headers, body = Handler.calls[-1]
    assert path == "/v1/systemone" and headers["Authorization"] == "Bearer k"
    assert body["model"] == "jev" and body["questions"]["action"]["type"] == "choice" and body["questions"]["action"]["criteria"] == {"hit": "hit", "stand": "stand"}


def test_decision_backend_runs_prompt_score_in_env(server):
    model = load_model({"backend": "decision", "id": "kev-latest", "base_url": server, "params": 9_000_000_000, "family": "kev"})
    assert model.info.params == 9_000_000_000 and "score" in model.capabilities and "generate" not in model.capabilities
    env = make_env("support", n_tickets=3)
    pol = make_policy("prompt_score", {}, model=model)
    pol.bind(env)
    ep, steps = run_episode(env, pol, seed=0)
    assert ep["steps"] == 3 and all(s["action"] == "deny" for s in steps)  # the fake prefers the last option
    assert steps[0]["extra"]["confidence"] == 0.8


def test_missing_option_gets_floor(server):
    m = DecisionAPIModel("jev", base_url=server)
    res = m.score_choices([Message("user", "u")], ["x"])
    assert res.logprobs[0] <= 0.0
