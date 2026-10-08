"""Loan approval: one-shot decisions from semantically meaningful features.

This is the "tabular decision" task. Each episode presents ``n_applicants`` independent applicants.
The hidden ground truth is a logistic default model with economically sensible signs (higher credit
score and longer employment lower default risk; higher debt ratio and loan-to-income raise it).
Reward for approving is ``+interest`` if the loan is repaid and ``-loss_given_default * amount`` on
default; denying yields 0. The oracle approves exactly when the true expected value is positive.
A distribution-shift switch supports out-of-distribution tests (H3):

* ``shift="none"``      : training distribution
* ``shift="covariate"`` : applicants drawn from a shifted feature distribution, same ground truth
* ``shift="sign_flip"`` : the credit-score effect is reversed, so semantic priors mislead
"""
from __future__ import annotations

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult

PURPOSES = ["car", "home improvement", "small business", "education", "debt consolidation"]
PURPOSE_RISK = {"car": -0.2, "home improvement": -0.3, "small business": 0.6, "education": 0.0, "debt consolidation": 0.4}


class LoanEnv(Environment):
    name = "loan"
    primary_metric = "profit"
    higher_is_better = True

    def __init__(self, n_applicants: int = 20, interest_rate: float = 0.12, loss_given_default: float = 0.6,
                 shift: str = "none", **params) -> None:
        super().__init__(n_applicants=n_applicants, interest_rate=interest_rate,
                         loss_given_default=loss_given_default, shift=shift, **params)
        self.n = int(n_applicants)
        self.interest_rate = float(interest_rate)
        self.lgd = float(loss_given_default)
        self.shift = shift
        self.action_labels = ["approve", "deny"]
        self.max_steps = self.n

    # ------------------------------------------------------------------ instance generation
    def _draw_applicants(self, rng: np.random.Generator) -> list[dict]:
        apps = []
        for _ in range(self.n):
            if self.shift == "covariate":
                income = float(np.clip(rng.lognormal(np.log(35), 0.5), 12, 400))
                credit = int(np.clip(rng.normal(600, 70), 300, 850))
            else:
                income = float(np.clip(rng.lognormal(np.log(55), 0.5), 12, 400))
                credit = int(np.clip(rng.normal(680, 70), 300, 850))
            apps.append({
                "age": int(rng.integers(21, 71)),
                "income_k": round(income, 1),
                "debt_ratio": round(float(np.clip(rng.beta(2, 5), 0, 0.9)), 2),
                "credit_score": credit,
                "years_employed": int(np.clip(rng.exponential(6), 0, 35)),
                "loan_k": round(float(np.clip(rng.lognormal(np.log(15), 0.6), 1, 150)), 1),
                "purpose": PURPOSES[int(rng.integers(len(PURPOSES)))],
            })
        return apps

    def _default_prob(self, a: dict) -> float:
        credit_sign = -1.0 if self.shift != "sign_flip" else 1.0
        z = (
            -1.2
            + credit_sign * 2.0 * (a["credit_score"] - 680) / 70
            + 2.5 * (a["debt_ratio"] - 0.28)
            + 1.2 * np.log1p(a["loan_k"] / a["income_k"])
            - 0.08 * min(a["years_employed"], 15)
            + PURPOSE_RISK[a["purpose"]]
        )
        return float(1.0 / (1.0 + np.exp(-z)))

    def _features(self, a: dict) -> np.ndarray:
        onehot = [1.0 if a["purpose"] == p else 0.0 for p in PURPOSES]
        return np.array([
            (a["age"] - 45) / 15, (a["income_k"] - 55) / 40, (a["debt_ratio"] - 0.28) / 0.15,
            (a["credit_score"] - 680) / 70, (a["years_employed"] - 6) / 6, (a["loan_k"] - 15) / 15,
        ] + onehot, dtype=np.float32)

    # ------------------------------------------------------------------ protocol
    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed)
        self.applicants = self._draw_applicants(self._rng)
        self.p_default = np.array([self._default_prob(a) for a in self.applicants])
        self.defaults = (self._rng.random(self.n) < self.p_default)
        self.decisions: list[str] = []
        self._t = 0
        self._done = False
        self._profit = 0.0
        self._regret = 0.0
        return self._obs()

    def _ev(self, i: int) -> float:
        a = self.applicants[i]
        gain = self.interest_rate * a["loan_k"]
        loss = self.lgd * a["loan_k"]
        p = self.p_default[i]
        return (1 - p) * gain - p * loss

    def step(self, action: str) -> StepResult:
        self._check_action(action, self.action_labels)
        i = self._t
        a = self.applicants[i]
        if action == "approve":
            r = -self.lgd * a["loan_k"] if self.defaults[i] else self.interest_rate * a["loan_k"]
        else:
            r = 0.0
        ev_approve = self._ev(i)
        oracle_value = max(ev_approve, 0.0)
        chosen_value = ev_approve if action == "approve" else 0.0
        self._regret += oracle_value - chosen_value
        self._profit += r
        self.decisions.append(action)
        self._t += 1
        self._done = self._t >= self.n
        return StepResult(self._obs(), float(r), self._done,
                          {"defaulted": bool(self.defaults[i]), "optimal": action == ("approve" if ev_approve > 0 else "deny")})

    def oracle_action(self) -> str:
        return "approve" if self._ev(self._t) > 0 else "deny"

    def episode_metrics(self) -> dict[str, float]:
        n_opt = sum(1 for i, d in enumerate(self.decisions) if d == ("approve" if self._ev(i) > 0 else "deny"))
        return {"profit": self._profit, "regret": self._regret, "optimal_decision_frac": n_opt / max(1, len(self.decisions)),
                "approve_rate": float(np.mean([d == "approve" for d in self.decisions])) if self.decisions else 0.0}

    def describe(self) -> str:
        return (
            "You are a loan officer deciding whether to approve loan applications. For each applicant you see their "
            "profile and the loan terms. If you approve and the loan is repaid, the bank earns the interest shown; "
            "if you approve and the applicant defaults, the bank loses the amount shown. Denying earns and loses "
            "nothing. Approve an application only when you expect it to be profitable on average, using the "
            "applicant's creditworthiness. Your goal is to maximise total profit."
        )

    def _obs(self) -> Observation:
        i = min(self._t, self.n - 1)
        a = self.applicants[i]
        text = (
            f"Applicant {self._t + 1} of {self.n}.\n"
            f"  Age: {a['age']}\n  Annual income: ${a['income_k']:.1f}k\n  Existing debt-to-income ratio: {a['debt_ratio']:.2f}\n"
            f"  Credit score: {a['credit_score']} (range 300-850)\n  Years in current employment: {a['years_employed']}\n"
            f"  Requested loan: ${a['loan_k']:.1f}k for {a['purpose']}\n"
            f"  If repaid the bank earns ${self.interest_rate * a['loan_k']:.1f}k; if the applicant defaults the bank "
            f"loses ${self.lgd * a['loan_k']:.1f}k."
        )
        return Observation(text=text, legal_actions=list(self.action_labels), features=self._features(a), info={"t": self._t})
