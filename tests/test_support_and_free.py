import numpy as np

from deadeye.envs import make_env
from deadeye.envs.support import SupportEnv
from deadeye.models.mock import MockModel
from deadeye.policies import make_policy
from deadeye.policies.prompts import PromptConfig, build_messages
from deadeye.runner import run_episode


def _ticket(**kw):
    t = {"issue": "damaged", "wants": "refund", "item": "headphones", "value": 120.0, "days_since_delivery": 3, "days_late": 0,
         "has_evidence": False, "prior_claims": 0, "premium": False, "in_stock": True,
         "message": "I received the package damaged. I need a refund today."}
    t.update(kw)
    return t


def test_support_rules_on_the_damaged_refund_example():
    assert SupportEnv.rule(_ticket()) == "request_evidence"            # expensive, no photo
    assert SupportEnv.rule(_ticket(has_evidence=True)) == "refund"      # photo attached, customer wants a refund
    assert SupportEnv.rule(_ticket(value=30.0)) == "refund"             # cheap: no evidence needed
    assert SupportEnv.rule(_ticket(prior_claims=3)) == "escalate"       # repeat claimer
    assert SupportEnv.rule(_ticket(wants="unspecified", has_evidence=True)) == "replace"
    assert SupportEnv.rule(_ticket(wants="unspecified", has_evidence=True, in_stock=False)) == "refund"
    assert SupportEnv.rule(_ticket(issue="changed_mind", days_since_delivery=20)) == "deny"
    assert SupportEnv.rule(_ticket(issue="changed_mind", days_since_delivery=20, premium=True)) == "refund"
    assert SupportEnv.rule(_ticket(issue="late", days_late=9)) == "partial_refund"
    assert SupportEnv.rule(_ticket(issue="not_received", days_late=10, value=250.0)) == "escalate"
    assert SupportEnv.rule(_ticket(issue="not_received", days_late=2)) == "deny"
    assert SupportEnv.rule(_ticket(issue="billing_error")) == "refund"


def test_support_rewards_punish_costly_mistakes():
    env = make_env("support", n_tickets=3)
    env.reset(0)
    env.tickets[0] = _ticket()  # oracle: request_evidence
    r_refund = env.step("refund").reward
    env.reset(0)
    env.tickets[0] = _ticket()
    r_deny = env.step("deny").reward
    env.reset(0)
    env.tickets[0] = _ticket()
    r_ok = env.step("request_evidence").reward
    assert r_ok == 1.0 and r_refund < r_deny < 0  # an unwarranted $120 refund costs more than a plain wrong answer


def test_support_env_text_contains_the_message_and_features():
    env = make_env("support", n_tickets=2)
    obs = env.reset(1)
    assert "Customer message:" in obs.text and obs.features.shape == (16,) and "escalate" in obs.legal_actions
    assert "Apply the company policy exactly" in env.describe()


def test_prompt_free_is_the_unconverted_condition():
    env = make_env("support", n_tickets=2)
    obs = env.reset(0)
    msgs, lm = build_messages(env, obs, [], PromptConfig(mode="free"))
    assert msgs[-1].content.endswith("What do you do? Answer in your own words.") and "Action: <" not in msgs[-1].content
    pol = make_policy("prompt_free", {}, model=MockModel(strategy="first"))
    assert pol.name == "prompt_free" and pol.max_new_tokens == 128 and pol.prompt.mode == "free"
    pol.bind(env)
    ep, steps = run_episode(env, pol, seed=0)
    assert ep["steps"] == 2 and all(s["action"] in s["legal_actions"] for s in steps)
