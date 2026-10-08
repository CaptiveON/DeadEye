# Known limitations and design decisions

Found during the adversarial review of the harness; kept here so the paper's limitations section
and the pre-registration stay honest. Items marked *decision* are deliberate and documented rather
than bugs.

## Environments

- **Tic-tac-toe oracle.** Against the random opponent the oracle is expectimax (the policy that
  maximises expected outcome against uniformly random replies); against the minimax opponent it is
  minimax. The minimax opponent breaks ties at random (seeded), so a deterministic policy meets many
  distinct games. Oracle agreement remains a lower bound wherever several moves are equally good.
- **Blackjack oracle.** Basic strategy for hit/stand with an infinite deck is the dynamic-programming
  optimum for these rules (expected value about -0.026 per hand), but the game has no doubling or
  splitting. Player and dealer cards come from separate pre-drawn streams, so pairing across policies
  is tight; the variance per hand is still large relative to the random-oracle span, which is why
  blackjack uses 300 episodes.
- **Contextual bandit and the feature-probe control.** The hidden weights are redrawn every seed and
  the feature vector is the current context only, so a classifier trained across seeds is at chance
  by construction. That control therefore measures nothing on this task; the task exists to test
  in-context learning, which the probe cannot do. *Decision.*
- **Few-shot demonstrations in tic-tac-toe** come from other seeds and may show the agent playing the
  other mark; each demonstration states its own mark in the observation text. *Decision.*

- **Clairvoyant bandit oracles.** On the bandit and contextual bandit the oracle knows the hidden
  means or weights, so it is an upper anchor rather than a policy any agent could implement from the
  text; UCB1 marks the achievable reference, and probes or LoRA trained on oracle labels learn a
  history-based approximation. *Decision.*
- **Blackjack episodes** can bundle several hands (`hands_per_episode`, 20 in the sweep) so that the
  per-episode normalised score has a workable variance; each hand is still a separate decision problem
  and the oracle is unchanged.

## Methods and runner

- **Training caps.** Probes and LoRA stop collecting states at `max_train_states` (default 50000);
  `prepare_stats` records `n_train_episodes_used` and `truncated` so a silently smaller data budget
  is visible in the logs.
- **Out-of-distribution loan tasks.** `train_params` lets adaptation methods train on the
  in-distribution variant and evaluate on the shifted one; `configs/ablations/tasks.yaml` uses it. Without
  it, probes and LoRA trained on the shifted task are not out of distribution.
- **Terminating on illegal actions** (`illegal_action: terminate`) cuts episodes short and makes
  regret look better for early failures; no shipped config uses it. *Decision.*
- **Served models.** An OpenAI-compatible server applies its own sampling defaults, and first-token
  scoring resolves ties at the floor in favour of the first option. Results from served models are
  reported separately and never enter the scale curves. *Decision.*

## Statistics

- Normalisation anchors (random and oracle means) are treated as fixed; their sampling uncertainty is
  not propagated into the confidence intervals. With 100 episodes it is small relative to the
  per-cell intervals, and the paired tests do not depend on the anchors at all.
- Holm correction is applied within one report call, which treats every comparison in that report as
  a single family; the pre-registration names the families per hypothesis and the paper applies the
  correction per family with `deadeye compare`.
