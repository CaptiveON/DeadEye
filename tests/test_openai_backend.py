"""Exercises the OpenAI-compatible client against a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from deadeye.models.base import Message
from deadeye.models.openai_backend import OpenAICompatModel


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        Handler.calls.append((self.path, body))
        if body.get("logprobs") and body["messages"][-1]["content"] == "words first":
            top = [{"token": "As", "logprob": -0.3}, {"token": "C", "logprob": -1.2}, {"token": "In", "logprob": -1.5},
                   {"token": "B.", "logprob": -2.0}, {"token": "A", "logprob": -2.5}, {"token": "I", "logprob": -3.0}]
            resp = {"choices": [{"message": {"content": "As"}, "logprobs": {"content": [{"token": "As", "logprob": -0.3,
                    "top_logprobs": top}]}}], "usage": {"prompt_tokens": 7, "completion_tokens": 1}}
        elif body.get("logprobs"):
            resp = {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"token": "B", "logprob": -0.1,
                    "top_logprobs": [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.5}, {"token": " C", "logprob": -4.0}]}]}}],
                    "usage": {"prompt_tokens": 42, "completion_tokens": 1}}
        else:
            resp = {"choices": [{"message": {"content": "Action: hit", "reasoning_content": "16 vs 10, must hit"}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 3}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_generate_and_score(server):
    m = OpenAICompatModel("fake-model", base_url=server, api_key="k", extra_body={"top_p": 0.9})
    msgs = [Message("system", "s"), Message("user", "Legal actions: hit, stand")]
    g = m.generate(msgs, max_new_tokens=5, stop=["\n"])
    assert g.text.startswith("<think>16 vs 10") and g.text.endswith("Action: hit") and g.prompt_tokens == 40
    path, body = Handler.calls[-1]
    assert path.endswith("/chat/completions") and body["top_p"] == 0.9 and body["stop"] == ["\n"] and body["max_tokens"] == 5
    s = m.score_choices(msgs, ["A", "B", "C", "D"])
    assert s.logprobs[1] == -0.1 and s.logprobs[0] == -2.5 and s.logprobs[2] == -4.0 and s.logprobs[3] < -4.0
    assert Handler.calls[-1][1]["logprobs"] is True and Handler.calls[-1][1]["max_tokens"] == 1


def test_unreachable_server_raises():
    m = OpenAICompatModel("x", base_url="http://127.0.0.1:9/v1", max_retries=0, timeout=1)
    with pytest.raises(RuntimeError):
        m.generate([Message("user", "hi")])


def test_first_token_scoring_ignores_words_starting_with_a_letter(server):
    """Tokens like "As" / "In" must not be credited to options "A" / "I"; "B." still counts for "B"."""
    m = OpenAICompatModel("fake-model", base_url=server)
    s = m.score_choices([Message("user", "words first")], list("ABCI"))
    assert s.logprobs == [-2.5, -2.0, -1.2, -3.0]
