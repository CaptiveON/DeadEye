"""Customer-support triage: decide what to do with a customer's message under an explicit company policy.

Each ticket is a short natural-language message ("I received the package damaged. I need a refund
today.") plus the order facts an agent would see (value, delivery timing, evidence, prior claims,
tier, stock). The company policy is stated in the task description, so the task tests whether a
model can apply written rules to a case: an instruction-following decision rather than a learned
one. The oracle applies the rules exactly. Rewards punish costly mistakes more than cheap ones.
"""
from __future__ import annotations

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult

ISSUES = ["damaged", "not_received", "wrong_item", "late", "changed_mind", "billing_error"]
WANTS = ["refund", "replacement", "unspecified"]
ITEMS = ["headphones", "coffee maker", "desk lamp", "running shoes", "backpack", "phone case", "blender", "jacket"]

MESSAGES: dict[str, dict[str, list[str]]] = {
    "damaged": {
        "refund": ["I received the package damaged. I need a refund today.",
                   "The {item} arrived with the box crushed and it does not work. Please refund my order.",
                   "My {item} came broken. I want my money back, not another one."],
        "replacement": ["The {item} arrived damaged, the casing is cracked. Can you send me a replacement?",
                        "Package was torn open and the {item} inside is scratched up. I'd like a new one sent out."],
        "unspecified": ["Just got my order and the {item} is damaged. What can you do about this?",
                        "The {item} was delivered damaged. Not happy about this."],
    },
    "not_received": {
        "refund": ["My order never arrived and the delivery date has passed. I want a refund.",
                   "Still no {item}. Tracking says nothing new. Refund me please."],
        "replacement": ["The {item} hasn't shown up. Can you ship another one?"],
        "unspecified": ["I haven't received my {item} yet. Where is it?",
                        "Order still missing. The promised date was days ago."],
    },
    "wrong_item": {
        "refund": ["You sent me the wrong product. I ordered a {item} and got something else. Refund me.",
                   "Wrong item in the box. I don't want it, please refund."],
        "replacement": ["I ordered a {item} but received a different product. Please send the correct one.",
                        "Got the wrong {item} model. Can you exchange it for the one I ordered?"],
        "unspecified": ["This isn't what I ordered. The box says {item} but it's a different product."],
    },
    "late": {
        "refund": ["My {item} arrived way after the promised date. I expect some money back for this."],
        "replacement": ["The {item} came late. Not sure what you can do but this was a gift and it missed the date."],
        "unspecified": ["The delivery was late. I did receive the {item} but this is poor service.",
                        "Order arrived after the delivery window. Disappointed."],
    },
    "changed_mind": {
        "refund": ["I changed my mind about the {item}. I'd like to return it for a refund.",
                   "I no longer need the {item}. How do I return it and get my money back?"],
        "replacement": ["I ordered the {item} but want a different colour instead. Can I swap it?"],
        "unspecified": ["I don't want the {item} anymore. It's unused."],
    },
    "billing_error": {
        "refund": ["I was charged twice for my {item} order. Please refund the duplicate charge.",
                   "My card shows two charges for one order. I want the extra one refunded."],
        "replacement": ["There's a wrong charge on my {item} order, please fix it."],
        "unspecified": ["The amount charged doesn't match the price I saw for the {item}."],
    },
}


class SupportEnv(Environment):
    name = "support"
    primary_metric = "optimal_decision_frac"
    higher_is_better = True

    def __init__(self, n_tickets: int = 10, **params) -> None:
        super().__init__(n_tickets=n_tickets, **params)
        self.n = int(n_tickets)
        self.action_labels = ["refund", "replace", "partial_refund", "request_evidence", "escalate", "deny"]
        self.max_steps = self.n

    # ------------------------------------------------------------------ instance
    def _draw_ticket(self, rng: np.random.Generator) -> dict:
        issue = ISSUES[int(rng.integers(len(ISSUES)))]
        wants = WANTS[int(rng.choice(3, p=[0.45, 0.25, 0.30]))]
        t = {
            "issue": issue, "wants": wants, "item": ITEMS[int(rng.integers(len(ITEMS)))],
            "value": float(np.round(np.clip(rng.lognormal(np.log(60), 0.8), 5, 500), 2)),
            "days_since_delivery": int(rng.integers(0, 61)),
            "days_late": int(rng.integers(0, 15)),
            "has_evidence": bool(rng.random() < 0.5),
            "prior_claims": int(rng.choice([0, 0, 0, 1, 1, 2, 3, 4])),
            "premium": bool(rng.random() < 0.3),
            "in_stock": bool(rng.random() < 0.7),
        }
        templates = MESSAGES[issue][wants]
        t["message"] = templates[int(rng.integers(len(templates)))].format(item=t["item"])
        return t

    @staticmethod
    def rule(t: dict) -> str:
        if t["prior_claims"] >= 3 and t["issue"] in ("damaged", "wrong_item", "not_received"):
            return "escalate"
        if t["issue"] == "billing_error":
            return "refund"
        if t["issue"] == "changed_mind":
            limit = 30 if t["premium"] else 14
            return "refund" if t["days_since_delivery"] <= limit else "deny"
        if t["issue"] == "late":
            return "partial_refund" if t["days_late"] > 7 else "deny"
        if t["issue"] == "not_received":
            if t["days_late"] <= 3:
                return "deny"
            return "escalate" if t["value"] > 100 else "refund"
        # damaged or wrong item
        if not t["has_evidence"] and t["value"] > 50:
            return "request_evidence"
        if t["wants"] == "refund":
            return "refund"
        return "replace" if t["in_stock"] else "refund"

    def _features(self, t: dict) -> np.ndarray:
        return np.array(
            [1.0 if t["issue"] == i else 0.0 for i in ISSUES] + [1.0 if t["wants"] == w else 0.0 for w in WANTS] +
            [t["days_since_delivery"] / 30, t["days_late"] / 14, t["value"] / 200, float(t["has_evidence"]),
             t["prior_claims"] / 3, float(t["premium"]), float(t["in_stock"])], dtype=np.float32)

    # ------------------------------------------------------------------ protocol
    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed + 30_013)
        self.tickets = [self._draw_ticket(self._rng) for _ in range(self.n)]
        self.decisions: list[str] = []
        self._t = 0
        self._done = False
        self._reward = 0.0
        self._payout_waste = 0.0
        return self._obs()

    def _reward_for(self, t: dict, action: str) -> float:
        oracle = self.rule(t)
        if action == oracle:
            return 1.0
        r = -1.0
        payout = {"refund": 1.0, "replace": 1.0, "partial_refund": 0.5}
        if action in payout and oracle in ("deny", "request_evidence", "escalate"):
            waste = payout[action] * t["value"] / 100
            self._payout_waste += waste
            r -= waste
        if action == "deny" and oracle in ("refund", "replace", "partial_refund"):
            r -= 1.0  # a legitimate claim refused: likely churn
        return r

    def step(self, action: str) -> StepResult:
        self._check_action(action, self.action_labels)
        t = self.tickets[self._t]
        r = self._reward_for(t, action)
        self._reward += r
        self.decisions.append(action)
        optimal = action == self.rule(t)
        self._t += 1
        self._done = self._t >= self.n
        return StepResult(self._obs(), r, self._done, {"optimal": optimal, "oracle": self.rule(t)})

    def oracle_action(self) -> str:
        return self.rule(self.tickets[self._t])

    def episode_metrics(self) -> dict[str, float]:
        n = max(1, len(self.decisions))
        opt = sum(1 for i, d in enumerate(self.decisions) if d == self.rule(self.tickets[i]))
        return {"optimal_decision_frac": opt / n, "payout_waste": self._payout_waste,
                "escalate_rate": float(np.mean([d == "escalate" for d in self.decisions])) if self.decisions else 0.0}

    def describe(self) -> str:
        return (
            "You are a customer support agent deciding how to resolve tickets. For each ticket you see the customer's "
            "message and the order facts. Apply the company policy exactly:\n"
            "1. If the customer has 3 or more prior claims and the issue is a damaged item, a wrong item or a missing "
            "delivery, escalate to the fraud team.\n"
            "2. Billing errors: refund.\n"
            "3. Changed mind: refund if the item was delivered at most 14 days ago (30 days for premium customers), "
            "otherwise deny.\n"
            "4. Late delivery (item received): partial_refund if it was more than 7 days late, otherwise deny.\n"
            "5. Not received: if it is at most 3 days past the promised date, deny (ask the customer to wait); if later "
            "and the order value is over $100, escalate; otherwise refund.\n"
            "6. Damaged or wrong item: if no photo evidence is attached and the order value is over $50, request_evidence; "
            "otherwise refund if the customer asks for a refund, else replace if a replacement is in stock, else refund.\n"
            "Correct decisions earn credit; unwarranted payouts cost the order value and refusing a legitimate claim loses "
            "the customer."
        )

    def _obs(self) -> Observation:
        i = min(self._t, self.n - 1)
        t = self.tickets[i]
        delivered = t["issue"] != "not_received"
        timing = (f"Delivered {t['days_since_delivery']} days ago" + (f", {t['days_late']} days after the promised date" if t["issue"] == "late" else "")
                  if delivered else f"Not delivered; {t['days_late']} days past the promised date")
        text = (f"Ticket {self._t + 1} of {self.n}.\n"
                f"Customer message: \"{t['message']}\"\n"
                f"Order: {t['item']}, value ${t['value']:.2f}. {timing}.\n"
                f"Photo evidence attached: {'yes' if t['has_evidence'] else 'no'}. Prior claims by this customer: {t['prior_claims']}. "
                f"Customer tier: {'premium' if t['premium'] else 'standard'}. Replacement in stock: {'yes' if t['in_stock'] else 'no'}.")
        return Observation(text=text, legal_actions=list(self.action_labels), features=self._features(t), info={"t": self._t})
