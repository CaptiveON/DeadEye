import pytest

from deadeye.envs import make_env
from deadeye.policies.parsing import parse_action, strip_think
from deadeye.policies.prompts import PromptConfig, build_messages, collect_demos, label_map, legal_block


@pytest.mark.parametrize("text,labels,want", [
    ("Action: up", ["up", "down"], "up"),
    ("I think...\nAction: `down`.", ["up", "down"], "down"),
    ("<think>maybe up</think>Action: left", ["up", "left"], "left"),
    ("<think>still thinking about up", ["up", "left"], None),
    ("Action: arm_3", ["arm_1", "arm_2", "arm_3"], "arm_3"),
    ("Action: B", ["A", "B", "C"], "B"),
    ("A good choice would be C.\nAction: (C)", ["A", "B", "C"], "C"),
    ("Because A is bad, B.", ["A", "B"], None),
    ("approve", ["approve", "deny"], "approve"),
    ("Action: approve the loan", ["approve", "deny"], "approve"),
    ("Action: stand.", ["hit", "stand"], "stand"),
    ("The best move is cell 5.", ["1", "5", "9"], None),
    ("ACTION = hit", ["hit", "stand"], "hit"),
    ("", ["hit", "stand"], None),
])
def test_parse_action(text, labels, want):
    got, _ = parse_action(text, labels)
    assert got == want


def test_strip_think_reports_text():
    clean, think = strip_think("<think>abc</think>Action: up")
    assert clean == "Action: up" and think == "abc"


def test_label_maps_and_blocks():
    assert label_map(["up", "down"], "name") == {"up": "up", "down": "down"}
    assert label_map(["up", "down"], "letter") == {"A": "up", "B": "down"}
    assert "Legal actions: A, B" in legal_block(["up", "down"], "letter")
    assert legal_block(["up", "down"], "name") == "Legal actions: up, down"
    with pytest.raises(ValueError):
        label_map(["x"], "emoji")


def test_build_messages_structure_and_fewshot():
    env = make_env("gridworld", size=4, min_distance=2)
    obs = env.reset(0)
    cfg = PromptConfig(action_format="letter", few_shot=2, cot=True)
    cfg.demos = collect_demos(lambda: make_env("gridworld", size=4, min_distance=2), list(range(100000, 100004)), 2)
    msgs, lm = build_messages(env, obs, [], cfg)
    assert msgs[0].role == "system" and env.describe() in msgs[0].content
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert all(v in env.action_labels for v in lm.values())
    assert "Legal actions:" in msgs[-1].content and "Action: <letter>" in msgs[-1].content
    assert msgs[2].content.splitlines()[-1].startswith("Action: ")


def test_score_mode_instruction():
    env = make_env("blackjack")
    obs = env.reset(0)
    msgs, _ = build_messages(env, obs, [], PromptConfig(mode="score"))
    assert msgs[-1].content.endswith("Answer with only the action name of your chosen action.")
