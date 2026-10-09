# Compute budget (one Apple M2 Max, 32 GB)

The whole study runs on one laptop through PyTorch's Metal backend at 16-bit precision. The numbers below are
estimates for planning; the pilot measures the real per-decision latency of every model (`latency_per_decision_s`
in each `summary.json`), and `deadeye estimate <config> --seconds-per-decision <measured>` turns it into hours.

| Model size | Scoring or probe (one forward pass) | Generation, 16 tokens | Free reply, 96 tokens |
|---|---|---|---|
| 135M to 600M | 0.05 to 0.1 s | 0.3 to 0.5 s | 2 to 3 s |
| 1.5B to 1.7B | 0.1 to 0.2 s | 0.5 to 0.8 s | 3 to 5 s |
| 3B to 4B | 0.3 s | 1 to 1.5 s | 6 to 10 s |
| 7B to 8B | 0.6 to 1 s | 1.5 to 2.5 s | 10 to 15 s |

Decisions per model and method in the main sweep are about 17,000 (bandit 5,000; contextual bandit 3,000; loan 2,000;
gridworld about 800; tic-tac-toe about 1,200; blackjack about 2,800; support 2,000).

| Stage | Config | Rough laptop time |
|---|---|---|
| Pilot | `configs/pilot.yaml` | 3 to 6 hours |
| Main sweep (28 checkpoints, 4 methods) | `configs/mac_main.yaml` | 150 to 200 hours, i.e. two to three weeks of overnight runs |
| LoRA (24 checkpoints to 4B) | `configs/mac_lora.yaml` | 50 to 80 hours |
| Free replies (12 checkpoints, 4 tasks, 50 episodes) | `configs/mac_free_reply.yaml` | 30 to 40 hours |
| Controls (Pythia to 2.8B, OLMo 2 to 7B) | `configs/mac_controls.yaml` | about 30 hours |
| Ablations (five blocks plus the LoRA companion, 50 episodes) | `configs/ablations/*.yaml` | about 100 hours in total |
| Quantisation block (llama.cpp servers) | `configs/ablations/quantisation.yaml` | about 15 hours |
| Decision models (local) | `configs/decision_models.yaml` | about 10 hours |

Everything is resumable, and `scripts/run_mac.sh` runs the stages in order while keeping the Mac awake. Memory: 16-bit
weights need about 2 bytes per parameter, so an 8B model takes about 16 GB plus a few GB of working memory; close other
applications while the 7B and 8B rungs run.
