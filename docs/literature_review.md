# DeadEye: annotated literature review

*Compiled 2026-10-08 for the working paper "Do small open-weight language models make good decision-makers? A controlled study of scale, conversion method and task structure."*

Companion files:

- `paper/refs.bib` holds 106 verified BibTeX entries. Keys are given in brackets below, e.g. `[krishnamurthy2024explore]`.
- `docs/model_ladders.md` holds the verified open-weight model ladders.

**Verification standard.** Every reference below was checked against at least one search result or page: an arXiv listing, a proceedings page (PMLR, NeurIPS, ICLR, ACL Anthology, AAAI), a publisher record, an official GitHub repository or a vendor announcement. The checks covered title, authors, year, venue and arXiv id. Where only the leading authors could be confirmed, the `.bib` entry ends with `and others`. Anything that could not be verified is listed in Section 4 and is not in the `.bib`. Annotations give 2-4 sentences: what the work did, what it found, and how it bears on hypotheses H1-H7.

**Hypotheses (pre-registered)**

- **H1:** decision quality rises roughly log-linearly with parameters under prompting, with a task-dependent slope.
- **H2:** instruction-tuned models beat base models under prompting; the gap vanishes under probe/LoRA.
- **H3:** a fine-tuned small model matches a 10x larger zero-shot model in-distribution but not out-of-distribution.
- **H4:** logit scoring removes format failures and closes much of the small-model gap.
- **H5:** 4-bit quantisation is nearly free above ~3B but costly below 1B.
- **H6:** exploration in bandits is poor at all scales without scaffolding.
- **H7:** a thinking/reasoning budget helps planning tasks, can hurt small models, and hurts latency-normalised efficiency.

### Hypothesis-to-evidence map (quick index)

| Hypothesis | Strongest prior evidence (supporting / complicating) |
|---|---|
| H1 scale, log-linear | Supporting: (Kaplan et al., 2020), (Brown et al., 2020), (Sinha et al., 2026), (Schmied et al., 2026). Complicating: (Wei, Tay, Bommasani, Raffel, Zoph, Borgeaud, Yogatama, Bosma, Zhou, Metzler, Chi, Hashimoto, Vinyals, Liang, Dean and Fedus, 2022), (Schaeffer et al., 2023), (Hoffmann et al., 2022) (tokens-per-parameter confound), (Jiwatode et al., 2026) |
| H2 base vs instruct | (Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, Lewis, Zettlemoyer and Levy, 2023), (Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula and Choi, 2024), (Wang, Ma, Hu, Weber-Genzel, Röttger, Kreuter, Hovy and Plank, 2024), (Wang, Hu, Ma, Röttger and Plank, 2024), (Kadavath et al., 2022), (Zhou, Lu, Mishra, Brahma, Basu, Luan, Zhou and Hou, 2023) |
| H3 fine-tuned small vs 10x zero-shot, ID vs OOD | (Dilkes et al., 2025), (Nie et al., 2025), (Chu et al., 2025), (Tajwar et al., 2025), (Ross et al., 2011), (Szot et al., 2024), (Binz et al., 2025) |
| H4 logit scoring | (Holtzman et al., 2021), (Robinson and Wingate, 2023), (Zheng et al., 2024), (Tan et al., 2024), (Carta et al., 2023), (Huang et al., 2022), (Sclar et al., 2024), (Tam et al., 2024) |
| H5 4-bit quantisation | (Dettmers and Zettlemoyer, 2023), (Kumar et al., 2025), (Zheng et al., 2026), (Liu, Sun, Zhang, Bai, Yu, Yu, Yuan and Hou, 2025), (Frantar et al., 2023), (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan and Han, 2024), (Dettmers et al., 2023) |
| H6 exploration | (Krishnamurthy et al., 2024), (Schmied et al., 2026), (Harris and Slivkins, 2026), (Nie et al., 2025), (Monea et al., 2025), (Binz and Schulz, 2023), (Laskin et al., 2023) |
| H7 thinking budget | (Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le and Zhou, 2022), (Liu, Geng, Wu, Sucholutsky, Lombrozo and Griffiths, 2025), (Shojaee et al., 2025), (Cuadron et al., 2025), (Gema et al., 2025), (Li et al., 2025), (Snell et al., 2025), (Jiwatode et al., 2026) |

---

## 1. Framing

Language models are now routinely deployed as decision-makers: agents that pick actions in games, tools and workflows. Most evidence about how well they do comes from three kinds of study:

- benchmark suites dominated by frontier API models (Paglieri et al., 2025; Ruoss et al., 2025; Liu et al., 2024);
- single-family studies with two or three sizes (Schmied et al., 2026; Jiwatode et al., 2026);
- methods papers that convert one model into a policy in one way: prompting (Yao et al., 2023), action-likelihood scoring (Carta et al., 2023; Tan et al., 2024), probing (Orgad et al., 2025), behaviour cloning (Chu et al., 2025) or RL (Wang et al., 2025; Liu et al., 2026).

Classical scaling work tells us loss falls smoothly with parameters (Kaplan et al., 2020; Hoffmann et al., 2022). It does not tell us whether *decision quality* does, or whether apparent jumps are artefacts of the metric (Wei, Tay, Bommasani, Raffel, Zoph, Borgeaud, Yogatama, Bosma, Zhou, Metzler, Chi, Hashimoto, Vinyals, Liang, Dean and Fedus, 2022; Schaeffer et al., 2023). Meanwhile, small open-weight models (sub-1B to ~14B) are promoted for agentic use on cost and latency grounds (Belcak et al., 2025). Recent work shows they can be greedy explorers (Schmied et al., 2026; Krishnamurthy et al., 2024), brittle to output format (Sclar et al., 2024; Wang, Ma, Hu, Weber-Genzel, Röttger, Kreuter, Hovy and Plank, 2024) and sensitive to quantisation (Kumar et al., 2025). When they "think", they can be helped or harmed depending on task and size (Li et al., 2025; Liu, Geng, Wu, Sucholutsky, Lombrozo and Griffiths, 2025; Shojaee et al., 2025).

DeadEye fills the space between these literatures. It uses same-recipe model ladders, several conversion methods (prompt-generate, prompt-score, probe, LoRA-SFT, optionally DPO/GRPO) and procedurally generated tasks with known oracles. Scores are normalised between random and oracle policies and reported with RL-grade statistics (Agarwal et al., 2021).

---

## 2. Themed review

### 2.1 Benchmarks of LLMs as agents and decision-makers

- **Wu, Tang, Mitchell & Li (2024), ICLR** `[wu2024smartplay]`
  - SmartPlay is a benchmark of six games (including Rock-Paper-Scissors, Tower of Hanoi and Minecraft, with a two-armed bandit among them). The games probe nine agentic capabilities such as planning, spatial reasoning, learning from history and understanding randomness, with procedurally varied instances.
  - It showed large capability gaps between models and across capabilities rather than a single "agent skill".
  - **Bearing:** this supports measuring *task-dependent* slopes (H1), and its bandit/randomness probes are a precedent for H6. DeadEye adds oracle-normalised scores and controlled within-family ladders, which SmartPlay lacked.

- **Paglieri, Cupiał, Coward et al. (2025), ICLR** `[paglieri2025balrog]`
  - BALROG evaluates LLMs and VLMs on games of graded difficulty (BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, NetHack).
  - Models made partial progress on easy environments but failed badly on the hardest. Adding visual input often hurt rather than helped.
  - **Bearing:** this is the clearest evidence that agentic difficulty is task-structured (H1 slope heterogeneity). Its model set is not a controlled ladder, so it cannot separate scale from recipe; that is the gap DeadEye's ladders target.

- **Ruoss, Pardo, Chan et al. (2025), ICML** `[ruoss2025lmact]`
  - LMAct tests in-context imitation learning with up to 512 expert episodes (up to ~1M tokens) on tic-tac-toe, chess, Atari, grid-world navigation, crosswords and a simulated cheetah, using frontier models.
  - Models seldom reach expert level, and more demonstrations often change little. Text vs image encoding and chain-of-thought matter.
  - **Bearing:** this is a direct precedent for our tic-tac-toe and gridworld tasks. It suggests in-context demonstrations are a weak substitute for weight updates, motivating H3's comparison of LoRA-SFT against prompting.

- **Liu, Yu, Zhang et al. (2024), ICLR** `[liu2024agentbench]`
  - AgentBench evaluates 29 API and open models in 8 interactive environments (OS, database, knowledge graph, games, web and others).
  - There is a wide gap between commercial and open models. The authors identify poor long-term reasoning, decision-making and instruction following as the main obstacles.
  - **Bearing:** the instruction-following failure mode is exactly what H4 posits logit scoring should remove. Separating format failures from decision failures is a methodological contribution DeadEye can make explicit.

- **Duan, Zhang, Diffenderfer et al. (2024), NeurIPS** `[duan2024gtbench]`
  - GTBench covers 10 game-theoretic tasks (complete vs incomplete information, deterministic vs probabilistic), played LLM-vs-LLM and LLM-vs-solver.
  - LLMs fail in complete-information deterministic games (the class containing tic-tac-toe) but are competitive in probabilistic ones. Code pre-training helps, and CoT/ToT do not always help.
  - **Bearing:** this predicts tic-tac-toe will be among our hardest tasks for small models (H1 slope). It also gives early evidence for H7's "reasoning is not uniformly beneficial".

- **Costarelli, Allen, Hauksson et al. (2024), arXiv** `[costarelli2024gamebench]`
  - GameBench evaluates strategic reasoning of LLM agents on a set of games chosen to be under-represented in pre-training data, comparing base prompting with reasoning scaffolds.
  - Tested models remained below the human baseline, and scaffolding changed scores materially.
  - **Bearing:** this supports treating scaffolding (CoT, summaries) as an explicit factor rather than a nuisance (H6, H7). It also illustrates contamination-aware game selection, relevant because tic-tac-toe and blackjack strategies are widely published online.

- **Guertler, Cheng, Yu et al. (2025), arXiv** `[guertler2025textarena]`
  - TextArena is an open-source collection of 57+ competitive text games (single-, two- and multi-player) with online TrueSkill ratings against humans and models, designed to be extensible for training as well as evaluation.
  - **Bearing:** this provides off-the-shelf adversarial environments and opponents for our tic-tac-toe task. It is also the substrate used for RL training in SPIRAL (Liu et al., 2026), useful for the DPO/GRPO stretch conversion.

- **Huang, Abbeel, Pathak & Mordatch (2022), ICML** `[huang2022zeroshot]`
  - Large, appropriately prompted LMs can decompose high-level tasks into plausible step-by-step plans in VirtualHome. However, the raw generations are often not admissible actions.
  - Mapping generations onto the admissible action set greatly improves executability at some cost in correctness.
  - **Bearing:** this is the canonical demonstration of the generate-then-parse problem that prompt-score avoids by construction (H4). Their executability/correctness trade-off suggests we should report parse/format failures separately from wrong-but-legal actions.

- **Yao, Zhao, Yu et al. (2023), ICLR** `[yao2023react]`
  - ReAct interleaves reasoning traces with actions. With only one or two in-context examples it beat imitation and RL baselines on ALFWorld and WebShop by 34 and 10 absolute points.
  - **Bearing:** this is the reference prompt-generate-with-reasoning baseline for planning tasks (H7). Its evidence comes from large models, and whether the benefit survives at 0.5-3B is exactly what H7 asks.

- **Shinn, Cassano, Berman et al. (2023), NeurIPS** `[shinn2023reflexion]`
  - Reflexion improves agents across trials by storing verbal self-reflections on feedback in an episodic memory, without weight updates. It gave gains on sequential decision-making (ALFWorld), reasoning and coding.
  - **Bearing:** this is a scaffold that converts reward history into usable text, analogous to the history summarisation that rescued exploration in (Krishnamurthy et al., 2024) (H6). It is a candidate "scaffolded" condition for bandits.

- **Jiwatode, Fuchs, Schmöcker et al. (2026), arXiv (for IEEE CoG 2026)** `[jiwatode2026spatial]`
  - This study crosses Qwen3 model scale, reasoning mode (thinking on/off) and planning horizon on a GVGAI benchmark of three spatial-navigation games with five difficulty levels.
  - Larger models with thinking enabled localise themselves more accurately, coordinate tracking stays limited for small models, and win rates fall with layout complexity. Adding causal context to prompts helps, especially for larger models.
  - **Bearing:** this is the closest published design to our gridworld × H1 × H7 cell. DeadEye extends it to several families, conversion methods, normalised scores and latency-normalised efficiency.

### 2.2 Exploration and bandits

- **Krishnamurthy, Harris, Foster, Zhang & Slivkins (2024), NeurIPS** `[krishnamurthy2024explore]`
  - The authors deployed GPT-3.5, GPT-4 and Llama 2 as agents in multi-armed bandits, with the interaction history in the prompt and many prompt designs.
  - Only GPT-4 with chain-of-thought *and* an externally summarised history (sufficient statistics) explored robustly. All other configurations failed.
  - **Bearing:** this is the foundational evidence for H6. It identifies history summarisation as the scaffold to include, and leaves open how failure rates vary across a parameter ladder.

- **Nie, Su, Chang et al. (2025), ICML** `[nie2025evolve]`
  - EVOLvE is a suite of context-free and contextual bandits that relates LLM regret to model size. It tests algorithm-guided inference-time support and algorithm distillation (in-context demonstrations and fine-tuning on trajectories generated by classic bandit algorithms).
  - With these interventions, smaller models achieved better exploration than larger unaided models on several tasks.
  - **Bearing:** this directly supports H3 in the bandit setting and refines H6: exploration is poor *without* scaffolding or distillation. It also implies that the LoRA-SFT teacher for bandits should be a learnable algorithm (UCB/Thompson), not the clairvoyant oracle.

- **Schmied, Bornschein, Grau-Moya, Wulfmeier & Pascanu (2026), ICLR** `[schmied2026greedy]`
  - The authors studied open models from 2B to 27B on multi-armed bandits, contextual bandits and tic-tac-toe, isolating three failure modes:
    - greediness (up to 55% of the action space left unexplored);
    - frequency bias (the 2B model copies the most frequent action in context, largely gone at 27B);
    - a knowing-doing gap (rationales correct ~87% of the time, yet the action does not follow).
  - RL fine-tuning on self-generated chain-of-thought with environment reward increased exploration and narrowed the gap.
  - **Bearing:** this is the strongest recent support for H6 (greediness persists with scale) and a scale-dependent failure relevant to H1. The knowing-doing gap is a mechanism by which probes could outperform generation (H2/H4). It also directly motivates our GRPO stretch.

- **Harris & Slivkins (2026), UAI** `[harris2026explore]`
  - The authors evaluated LLMs on exploration and exploitation separately in (contextual) bandits.
  - LLMs often struggle even to *exploit*, falling short of simple linear regression despite in-context mitigations. Reasoning models are most promising but too slow or expensive for many settings. LLMs help exploration mainly by proposing candidates in large, semantically meaningful action spaces.
  - **Bearing:** this is relevant to H6 and to our contextual-bandit and loan tasks: decision failures may be exploitation failures, not only exploration failures. The latency caveat supports H7's efficiency clause.

- **Monea, Bosselut, Brantley & Artzi (2025), COLM** `[monea2025icrl]`
  - The authors studied in-context RL framed as a contextual bandit on classification tasks, for Llama 3.1, Qwen2.5 and Gemini 1.5 Flash from 500M to 70B parameters.
  - Models can learn online from external reward in context. An earlier version of the paper reported that naive in-context RL fails because of insufficient exploration.
  - **Bearing:** this is a rare multi-size (500M-70B) result on reward-driven in-context learning, closely matching our contextual-bandit task. It is evidence for H1 (size effects) and H6 (exploration as the bottleneck).

- **Tajwar, Jiang, Thankaraj et al. (2025), arXiv** `[tajwar2025curious]`
  - PAPRIKA fine-tunes LLMs on synthetic interaction trajectories from diverse decision tasks (e.g., twenty questions) with a curriculum, so that models learn to gather information in context.
  - Fine-tuned models transfer improved decision-making to *unseen* task families without further training.
  - **Bearing:** this cuts against the OOD half of H3: fine-tuning on a *diverse* task distribution can generalise. DeadEye's OOD splits should therefore distinguish within-task shift from across-task shift.

- **Laskin, Wang, Oh et al. (2023), ICLR** `[laskin2023incontext]`
  - Algorithm Distillation trains a causal transformer on the learning histories of an RL algorithm. The model then performs RL in context, more data-efficiently than the source algorithm, across sparse-reward, combinatorial and pixel-based tasks.
  - **Bearing:** this is the theoretical backdrop for H6 and H3. Behaviour cloning on *oracle* trajectories (which never explore) cannot teach exploration, whereas cloning *learning histories* can.

- **Auer, Cesa-Bianchi & Fischer (2002), Machine Learning** `[auer2002ucb]`
  - This paper introduced UCB1 and showed that logarithmic regret is achievable uniformly over time for bounded rewards.
  - **Bearing:** it supplies the standard non-LLM reference policy for bandits. In our normalisation it is a strong "competent algorithm" anchor between random and the clairvoyant oracle (H6).

- **Lattimore & Szepesvári (2020), Cambridge University Press** `[lattimore2020bandit]`
  - This is the standard monograph on bandits: regret definitions, lower bounds, UCB/Thompson sampling, and linear/contextual bandits.
  - **Bearing:** we use it for regret-based score definitions and LinUCB/Thompson baselines in the contextual bandit (H6), and for specifying what an "oracle" means in each bandit task.

### 2.3 Scale, emergence and small models

- **Kaplan, McCandlish, Henighan et al. (2020), arXiv** `[kaplan2020scaling]`
  - Cross-entropy loss follows power laws in non-embedding parameters, data and compute over more than seven orders of magnitude, with architecture details mattering little.
  - **Bearing:** this motivates the log-linear form of H1 and the use of *non-embedding* parameters as the x-axis. For sub-1B models with large vocabularies the embedding share is very large (e.g., ~170M of Gemma 3 270M's parameters), so the choice of x-axis materially changes H1's slope.

- **Hoffmann, Borgeaud, Mensch et al. (2022), NeurIPS** `[hoffmann2022chinchilla]`
  - From 400+ training runs, the authors concluded that parameters and training tokens should scale roughly equally. Chinchilla (70B, ~4x more data) beat the 280B Gopher.
  - **Bearing:** this is a key confound for H1. Within modern ladders, small rungs are trained far beyond compute-optimal token counts (SmolLM2-1.7B on ~11T tokens (Ben Allal et al., 2025)), so "parameters" partly proxies for tokens-per-parameter. That also matters for H5 via (Kumar et al., 2025).

- **Wei, Tay, Bommasani et al. (2022), TMLR** `[wei2022emergent]`
  - The authors catalogued "emergent" abilities that are near-chance in small models and appear abruptly in large ones, so they cannot be extrapolated from small-model trends.
  - **Bearing:** this is the main alternative to H1's log-linear form. Planning-heavy tasks (gridworld, tic-tac-toe) could show threshold behaviour rather than smooth slopes.

- **Schaeffer, Miranda & Koyejo (2023), NeurIPS** `[schaeffer2023mirage]`
  - Apparent emergence is often produced by nonlinear or discontinuous metrics (exact-match accuracy) and small test sets. Continuous metrics yield smooth, predictable scaling.
  - **Bearing:** this is a methodological constraint on H1. We should fit scaling on continuous normalised scores (regret, return) and report thresholded success separately, with enough episodes per cell.

- **Brown, Mann, Ryder et al. (2020), NeurIPS** `[brown2020gpt3]`
  - GPT-3 (175B) and a within-family ladder of smaller models showed that few-shot in-context learning improves strongly with scale, without gradient updates.
  - **Bearing:** this is the original same-recipe ladder evidence for H1 in the prompting regime. It is also the premise of our contextual-bandit and loan tasks, which require learning from in-context examples.

- **Sinha, Arun, Goel, Staab & Geiping (2026), ICLR** `[sinha2026illusion]`
  - The study isolates *execution* from planning by supplying the plan and knowledge, then measures how many steps models can execute.
  - Scaling model size keeps improving the achievable horizon with non-diminishing returns, even when single-step accuracy already looks saturated.
  - **Bearing:** this supports H1 and implies the H1 slope should increase with horizon length. DeadEye should therefore sweep episode length in gridworld and report per-step as well as per-episode metrics.

- **Belcak, Heinrich, Diao et al. (2025), arXiv (position paper, NVIDIA)** `[belcak2025slm]`
  - The authors argue that small language models are sufficiently capable, better suited and more economical for most invocations in agentic systems. They propose an LLM-to-SLM agent conversion procedure and heterogeneous systems where needed.
  - **Bearing:** this provides the policy motivation for the study. H3 (specialised small models) and H7's latency-normalised efficiency are the empirical tests of its claims.

- **Dilkes, Yazdanpanah & Stein (2025), arXiv** `[dilkes2025reinforced]`
  - The authors introduced multi-step GRPO (MS-GRPO), which credits episode reward to every step, plus advantage-weighted episode sampling.
  - A post-trained 3B model beat a 72B baseline by 50% on Frozen Lake (also tested on Snake).
  - **Bearing:** this is direct in-distribution evidence for H3 (a >10x gap closed by post-training), but without OOD tests, which is the half of H3 DeadEye adds. It is also a template for our GRPO stretch.

- **Snell, Lee, Xu & Kumar (2025), ICLR** `[snell2025ttc]`
  - Allocating test-time compute adaptively per prompt (verifier search or revision) is >4x more efficient than best-of-N. In FLOPs-matched comparisons, a small model plus test-time compute can beat a 14x larger model on problems where it has non-trivial success.
  - **Bearing:** this frames H7 as a parameters-vs-inference-compute trade-off and motivates reporting FLOP- or latency-matched comparisons alongside raw scores.

### 2.4 Conversion methods: prompting, scoring, probing, fine-tuning, RL

#### Prompt-generate and prompt-score

- **Holtzman, West, Shwartz, Choi & Zettlemoyer (2021), EMNLP** `[holtzman2021surface]`
  - Scoring answer strings by raw probability is distorted by "surface form competition": synonyms split probability mass.
  - Domain-conditional PMI scoring gives consistent zero-shot gains over calibrated and uncalibrated scoring for GPT-2/GPT-3.
  - **Bearing:** this is central to the prompt-score design (H4). Action labels ("left" vs "west", "hit" vs "draw another card") compete, so DeadEye should pre-register raw, length-normalised and PMI-style scores.

- **Robinson, Rytting & Wingate (2023), ICLR** `[robinson2023mcsb]`
  - The paper compares cloze-style scoring of each option with presenting all options jointly and scoring the option *symbol*. The joint format works only for models with strong "multiple choice symbol binding", an ability that varies greatly by model.
  - **Bearing:** this suggests two prompt-score variants: scoring action strings (robust for weak or small models) and scoring option letters (efficient but dependent on symbol binding). Their contrast across scale is a direct test of H4.

- **Zheng, Zhou, Meng, Zhou & Huang (2024), ICLR** `[zheng2024selectors]`
  - Across 20 LLMs, multiple-choice answers shift when option order changes because models prefer particular option IDs (a token-level prior). The label-free PriDe method estimates and removes this prior.
  - **Bearing:** letter-indexed action scoring will inherit ID/position bias, so action order must be randomised or debiased (H4). An apparent "preference" for one bandit arm could be ID bias rather than learned value (H6).

- **Wang et al. (2024), Findings of ACL** `[wang2024myanswerc]`
  - For instruction-tuned models, first-token option probabilities frequently disagree with the answer actually generated (over 60% mismatch for Llama-2-7B-Chat), due to conversational preambles and refusals. The mismatch persists under constrained prompts.
  - **Bearing:** this is a caution for H4 and H2. Logit scoring of instruct models must score full action strings under the chat template, and base-vs-instruct comparisons under scoring and generation can diverge for reasons unrelated to decision quality.

- **Wang, Hu, Ma, Röttger & Plank (2024), arXiv** `[wang2024lookattext]`
  - When first-token and text answers disagree often, the *text* answers of instruction-tuned models are more robust to perturbations than first-token probabilities, even after PriDe debiasing.
  - **Bearing:** this complicates H4. Scoring is not automatically more robust than generation for instruct models, so the expected benefit concentrates in base and small models whose generations fail to parse.

- **Zhao, Wallace, Feng, Klein & Singh (2021), ICML** `[zhao2021calibrate]`
  - Few-shot predictions are biased toward majority labels, recent labels and common tokens.
  - Contextual calibration, which fits a correction so that a content-free input ("N/A") yields uniform predictions, improves GPT-2/3 accuracy by up to 30 points.
  - **Bearing:** this is a cheap calibrated-scoring variant for H4. Recency and majority biases are a prompt-level explanation for "frequency bias" in bandit histories (H6).

- **Sclar, Choi, Tsvetkov & Suhr (2024), ICLR** `[sclar2024formatting]`
  - Meaning-preserving prompt-format changes shift few-shot accuracy by up to 76 points (LLaMA-2-13B). The sensitivity persists with scale, more shots and instruction tuning.
  - FormatSpread estimates the performance range across formats.
  - **Bearing:** DeadEye should report a format range, not a single prompt, per (model, task). The size of this spread versus scale is itself a test of H4 (scoring should shrink it).

- **Hegselmann, Buendia, Lang et al. (2023), AISTATS** `[hegselmann2023tabllm]`
  - TabLLM serialises tabular rows into natural language for LLM classification. It is non-trivial zero-shot and competitive with gradient-boosted trees in the very-few-shot regime after light fine-tuning, with serialisation choice mattering.
  - **Bearing:** this is the closest precedent for our semantic loan-approval task. It suggests prompt knowledge helps at low data while fine-tuning catches up quickly (H3), and that serialisation is a format factor (H4).

#### Probing frozen hidden states

- **Alain & Bengio (2016), arXiv** `[alain2016probes]`
  - Linear classifier probes are trained post hoc on frozen intermediate layers without affecting the model. In the vision networks studied, linear separability increased monotonically with depth.
  - **Bearing:** this defines our probe conversion: a linear head on frozen states, trained per layer. Whether separability peaks before the last layer in LLMs is addressed by (Skean et al., 2025).

- **Belinkov (2022), Computational Linguistics** `[belinkov2022probing]`
  - This review covers probing classifiers and their pitfalls: probe expressivity (linear vs MLP), the need for control tasks and baselines, and the gap between decodability and use by the model.
  - **Bearing:** it is essential for interpreting H2/H3 under the probe method. A probe that recovers the optimal action shows the information is *decodable*, not that the LM *uses* it. We need controls (randomly initialised model, raw-feature baselines) and should prefer linear over MLP probes when making claims.

- **Skean et al. (2025), ICML** `[skean2025layer]`
  - Across 32 embedding tasks, architectures and modalities, intermediate layers often give better features than the final layer, which can over-specialise to next-token prediction.
  - **Bearing:** the probe protocol must sweep layers. The best layer may shift with model size, a confound for H2's "gap vanishes under probe".

- **Orgad et al. (2025), ICLR** `[orgad2025know]`
  - Truthfulness signals concentrate in the exact-answer tokens, enabling better probe-based error detection.
  - Models often encode the correct answer internally while generating a wrong one. Error detectors do not generalise across datasets.
  - **Bearing:** this is evidence for a "knows more than it shows" gap that probes can exploit (H2, H4). The poor cross-dataset generalisation predicts probes will also degrade out of distribution (H3).

- **Kadavath, Conerly, Askell, Henighan et al. (2022), arXiv** `[kadavath2022know]`
  - Larger models are well calibrated on diverse multiple-choice and true/false questions when the format is right, and can estimate P(True) for their own answers, with self-evaluation improving with scale.
  - **Bearing:** this suggests action log-probabilities become more meaningful (calibrated) with scale, a mechanism behind H4. Format dependence links to H2.

#### Behaviour cloning, LoRA and alignment

- **Pomerleau (1988), NIPS** `[pomerleau1988alvinn]`
  - ALVINN trained a small network on simulated road images to output steering, then followed real roads under some conditions. It is the canonical early behaviour-cloning system.
  - **Bearing:** this is the historical root of our LoRA-SFT conversion and of the distribution-shift problem that motivates the OOD half of H3.

- **Ross, Gordon & Bagnell (2011), AISTATS** `[ross2011dagger]`
  - Plain behaviour cloning suffers compounding errors under its own state distribution. DAgger aggregates expert labels on learner-visited states, reducing imitation to no-regret online learning.
  - **Bearing:** this predicts LoRA-SFT on oracle trajectories degrades with horizon (gridworld) and under shift (H3). Because our oracles are cheap to query, a DAgger-style relabelling ablation is inexpensive.

- **Chen, Lu, Rajeswaran et al. (2021), NeurIPS** `[chen2021decision]`
  - Decision Transformer treats offline RL as return-conditioned sequence modelling with a GPT-style network, matching or beating offline RL baselines on Atari, Gym and Key-to-Door.
  - **Bearing:** return conditioning is an alternative to plain behaviour cloning for the fine-tuning conversion, especially with non-oracle data. It is the conceptual bridge between LLM fine-tuning and RL (H3).

- **Hu, Shen, Wallis et al. (2022), ICLR** `[hu2022lora]`
  - LoRA freezes pre-trained weights and trains low-rank update matrices, cutting trainable parameters by ~10,000x for GPT-3 with no inference latency, while matching full fine-tuning quality.
  - **Bearing:** this is our LoRA-SFT conversion. Rank and learning-rate choices must be held fixed or tuned per rung to avoid confounding H3 with optimisation differences.

- **Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), NeurIPS** `[dettmers2023qlora]`
  - QLoRA backpropagates through a frozen 4-bit (NF4, double-quantised) model into LoRA adapters, enabling 65B fine-tuning on one 48GB GPU with 16-bit-level quality.
  - **Bearing:** this makes LoRA feasible on the largest rungs and creates an H5 × H3 interaction (fine-tuning on a quantised base) that should be logged as a factor.

- **Szot, Schwarzer, Agrawal et al. (2024), ICLR** `[szot2024llarp]`
  - LLaRP keeps a pre-trained LLM frozen and learns observation and action adapters with online RL in an embodied rearrangement benchmark.
  - It generalised better than baselines to paraphrased instructions and novel task behaviours.
  - **Bearing:** this is evidence that LLM-initialised policies can generalise OOD (against a strong reading of H3's OOD clause). It is also an architectural precedent for adapter or probe heads on frozen LMs.

- **Chu, Zhai et al. (2025), ICML** `[chu2025sft]`
  - The authors compared SFT and RL post-training on a card-arithmetic game (GeneralPoints) and a navigation environment (V-IRL) with textual and visual rule variants.
  - SFT tends to memorise the training rules while outcome-reward RL generalises to unseen variants. SFT is still useful for stabilising the output format before RL.
  - **Bearing:** this is the most direct prior support for the OOD half of H3 (LoRA-SFT fails OOD) and motivates GRPO as the stretch comparison. The format-stabilisation role links to H4.

- **Zhou et al. (2023), NeurIPS** `[zhou2023lima]`
  - LIMA fine-tunes a 65B LLaMA on ~1,000 curated examples with no RLHF and is competitive with much more heavily aligned models.
  - The authors conclude that nearly all knowledge is learned in pre-training and instruction tuning mainly teaches output style and format.
  - **Bearing:** this is the theoretical basis for H2's second clause: if instruction tuning is mostly format, base-vs-instruct gaps should vanish under probe or LoRA.

- **Lin, Ravichander, Lu et al. (2024), ICLR** `[lin2024urial]`
  - Base and aligned models (e.g., Llama-2 vs Llama-2-chat) share top-ranked tokens at most positions, differing mainly on stylistic tokens.
  - URIAL, a tuning-free in-context alignment method (three stylistic examples plus a system prompt), lets base models match SFT/RLHF models.
  - **Bearing:** this is a strong challenge to H2's first clause. With a few-shot scaffold, base models might close the instruct gap even under prompting, so DeadEye needs a "base + URIAL-style few-shot" prompt condition.

#### RL and preference optimisation with LLM policies

- **Rafailov, Sharma, Mitchell et al. (2023), NeurIPS** `[rafailov2023dpo]`
  - DPO reparameterises the RLHF objective so the optimal policy is fit with a classification loss on preference pairs, matching or beating PPO-based RLHF with less complexity.
  - **Bearing:** this is the stretch conversion. Our oracles provide natural preference pairs (oracle-preferred vs dispreferred action), making DPO a cheap middle ground between SFT and online RL for H3.

- **Shao, Wang, Zhu et al. (2024), arXiv** `[shao2024deepseekmath]`
  - The paper introduced Group Relative Policy Optimization (GRPO), which estimates advantages from groups of sampled completions without a critic. GRPO plus targeted pre-training gave a 7B model 51.7% on competition MATH.
  - **Bearing:** this is the RL algorithm for our GRPO stretch with environment reward. Being critic-free makes it practical on small rungs.

- **DeepSeek-AI (2025), arXiv; peer-reviewed version in Nature (2025)** `[deepseekai2025r1]`
  - Reasoning behaviours (self-verification, reflection) emerged from large-scale RL without SFT (R1-Zero); multi-stage training gave R1. Six dense distilled models (1.5B-70B, on Qwen2.5 and Llama bases) were released.
  - **Bearing:** the distilled ladder offers an H7 contrast at fixed size: a reasoning-distilled checkpoint vs its non-reasoning base (see `model_ladders.md`).

- **Carta, Romac, Wolf et al. (2023), ICML** `[carta2023glam]`
  - GLAM uses Flan-T5 models of several sizes as policies in BabyAI-Text, computing each legal action's probability from the LM's token likelihoods and updating the LM online with PPO.
  - Grounding improved sample efficiency and some forms of generalisation.
  - **Bearing:** this is the clearest precedent for "prompt-score as a policy" (H4) and for RL on top of it. Its size comparisons are small and within one family, which DeadEye extends.

- **Tan, Zhang, Liu et al. (2024), ICLR** `[tan2024twosome]`
  - TWOSOME forms a policy from the joint probabilities of valid action strings. Longer actions are unfairly penalised, so the authors propose token- and word-level normalisation, then train with PPO in Overcooked and VirtualHome.
  - It beat PPO and SayCan in sample efficiency and performance.
  - **Bearing:** this is direct design guidance for H4: action-length normalisation must be part of prompt-score, or multi-token actions (e.g., "stand" vs "hit") will be systematically biased.

- **Wang, Wang et al. (2025), arXiv** `[wang2025ragen]`
  - RAGEN/StarPO is trajectory-level multi-turn RL for LLM agents, demonstrated on Sokoban, FrozenLake and a bi-arm bandit with Qwen2.5-0.5B-Instruct.
  - It identifies an "Echo Trap" collapse (reward-variance cliffs, gradient spikes, vanishing reasoning) and proposes a stabilised variant (StarPO-S). Diverse initial states and frequent sampling help.
  - **Bearing:** this is a practical warning for our GRPO stretch on tiny models (monitor reward variance and reasoning length). Its bandit results are relevant to H6 after RL.

- **Liu, Guertler, Yu et al. (2026), ICLR** `[liu2026spiral]`
  - SPIRAL trains via online multi-agent self-play on zero-sum text games (TicTacToe, Kuhn Poker, Simple Negotiation) with role-conditioned advantage estimation.
  - Qwen3-4B-Base trained on Kuhn Poker alone improved math (+8.6%) and general reasoning (+8.4%), beating SFT on 25k expert game trajectories.
  - **Bearing:** this shows RL on our kind of adversarial game is feasible at 4B and that game-trained skills can transfer (relevant to H3's OOD clause). Tic-tac-toe self-play is a ready-made stretch condition.

### 2.5 Reasoning/thinking and format effects

- **Wei, Wang, Schuurmans et al. (2022), NeurIPS** `[wei2022cot]`
  - Few-shot chain-of-thought exemplars greatly improve arithmetic, commonsense and symbolic reasoning in large models. The benefit appears only at large scale, and small models produce fluent but illogical chains.
  - **Bearing:** this is the origin of H7's "reasoning can hurt small models". Our prompt-generate CoT variant across rungs is a direct replication in decision tasks.

- **Wang, Wei, Schuurmans et al. (2023), ICLR** `[wang2023selfconsistency]`
  - Self-consistency samples multiple reasoning paths and takes a majority vote, giving large gains over greedy CoT (e.g., +17.9 points on GSM8K).
  - **Bearing:** this is a test-time-compute alternative to thinking modes for H7. Its cost must be counted in latency-normalised efficiency, and majority voting over actions is a natural ensemble policy for prompt-generate.

- **Liu, Geng, Wu, Sucholutsky, Lombrozo & Griffiths (2025), ICML** `[liu2025mindstep]`
  - CoT and reasoning models lose accuracy on tasks where verbal deliberation also hurts humans: implicit statistical learning, visual recognition and classification with exceptions (up to 36.3 points for o1-preview vs GPT-4o).
  - **Bearing:** this predicts thinking may *hurt* on our tasks that resemble implicit statistical learning from feedback (contextual bandits, loan approval), supporting the task-dependence in H7.

- **Shojaee, Mirzadeh, Alizadeh et al. (2025), NeurIPS** `[shojaee2025illusion]`
  - On controllable puzzles (e.g., Tower of Hanoi), matched thinking vs non-thinking models show three regimes: non-thinking is better at low complexity, thinking helps at medium complexity, and both collapse at high complexity. Reasoning models also fail to execute explicit algorithms reliably.
  - **Bearing:** this predicts a complexity-dependent sign for H7. Gridworld size and tic-tac-toe depth should be swept so that "thinking helps planning" is tested across the regimes.

- **Cuadron et al. (2025), arXiv** `[cuadron2025overthinking]`
  - On SWE-bench Verified (4,018 trajectories), reasoning models "overthink" (analysis paralysis, rogue actions, premature disengagement) about 3x more than non-reasoning models, and higher overthinking predicts lower success.
  - Selecting low-overthinking solutions improves accuracy while cutting cost.
  - **Bearing:** this is direct agentic evidence for H7's claim that thinking can hurt success and efficiency. It suggests logging reasoning length and interaction counts per episode.

- **Gema et al. (2025), TMLR** `[gema2025inverse]`
  - Constructed tasks (counting with distractors, regression with spurious features, deduction with constraint tracking, safety-relevant probes) show *inverse* scaling: longer reasoning lowers accuracy in large reasoning models.
  - **Bearing:** the "regression with spurious features" family closely resembles our loan-approval task. This predicts that a larger thinking budget can reduce accuracy there (H7), so budget sweeps rather than on/off toggles are needed.

- **Li et al. (2025), Findings of ACL** `[li2025smallstruggle]`
  - Models of ~3B or fewer do not reliably benefit from long CoT or from distillation from strong reasoners. They learn better from shorter chains matched to their capacity ("Mix Distillation").
  - **Bearing:** this is strong prior support for H7's "can hurt small models" and a caution for interpreting the small DeepSeek-R1 distills.

- **Tam, Wu, Tsai et al. (2024), EMNLP Industry Track** `[tam2024speakfreely]`
  - Forcing structured outputs (JSON/XML, format-restricting instructions) significantly degrades reasoning, more so with stricter constraints, though it can help on classification.
  - **Bearing:** this is relevant to H4 and H7. Strict output formats help parsing but may suppress reasoning, so prompt-generate should allow free reasoning followed by a delimited action, and constrained decoding should be a separate condition.

- **Zhou, Lu, Mishra et al. (2023), arXiv** `[zhou2023ifeval]`
  - IFEval measures instruction following with ~500 prompts built on 25 types of automatically verifiable instructions.
  - **Bearing:** published IFEval scores are a covariate to predict per-model format-failure rates under prompt-generate (H4) and to explain base-vs-instruct differences (H2).

### 2.6 Quantisation

- **Dettmers & Zettlemoyer (2023), ICML** `[dettmers2023kbit]`
  - More than 35,000 zero-shot experiments (19M-176B parameters; BLOOM, OPT, Pythia/NeoX, GPT-2) show 4-bit precision is almost universally optimal for accuracy at a fixed total-bit budget. Small block sizes and the right data type help.
  - **Bearing:** this is the baseline expectation for H5 (4-bit nearly free). It was measured on zero-shot language tasks and older, less over-trained models, which is exactly what our decision tasks and modern ladders test.

- **Frantar, Ashkboos, Hoefler & Alistarh (2023), ICLR (published as OPTQ; widely cited as GPTQ)** `[frantar2023gptq]`
  - One-shot second-order post-training quantisation compresses 175B GPT models to 3-4 bits in about four GPU hours with little accuracy loss, and gives reasonable accuracy even at 2-bit/ternary.
  - **Bearing:** this is one of our two 4-bit methods for H5. Its calibration-data dependence should be held fixed across rungs.

- **Lin, Tang, Tang et al. (2024), MLSys** `[lin2024awq]`
  - AWQ protects the ~1% of salient weight channels identified from *activation* statistics via per-channel scaling, giving accurate low-bit weight-only quantisation suitable for on-device inference.
  - **Bearing:** this is the second 4-bit method for H5. Agreement between GPTQ and AWQ penalties at each rung tests whether H5 is method-specific.

- **Kumar, Ankner, Spector et al. (2025), ICLR** `[kumar2025precision]`
  - Precision-aware scaling laws, fit on 465+ runs, show post-training-quantisation degradation *grows* with the amount of pre-training data, to the point that more data can hurt the quantised model. Low precision acts like a reduced "effective parameter count".
  - **Bearing:** this is the mechanism behind H5. Small modern models are the most over-trained, so the prediction is larger 4-bit penalties below ~1B. It also suggests tokens-per-parameter as a moderator in the H5 analysis.

- **Zheng, Li, Chu et al. (2025), arXiv** `[zheng2025qwen3quant]`
  - The authors applied RTN, GPTQ, AWQ, SmoothQuant and BiLLM to Qwen3 at 1-8 bits. Qwen3 stays competitive at moderate bit-widths but degrades sharply at ultra-low precision.
  - **Bearing:** this is same-family evidence for H5 on one of our candidate ladders, measured on perplexity/MMLU rather than decision quality.

- **Liu, Sun, Zhang et al. (2025), arXiv** `[liu2025quanthurts]`
  - On quantised reasoning models (DeepSeek-R1 distills 1.5B-70B, QwQ-32B, Qwen3-8B), W8A8 and W4A16 are near-lossless but lower bit-widths are risky. Degradation depends on model size, origin and task difficulty, and quantised models do not produce longer outputs.
  - **Bearing:** this is evidence for an H5 × H7 interaction. Thinking-mode accuracy at 4-bit should be tested separately from non-thinking accuracy, especially at small sizes.

### 2.7 Evaluation methodology and statistics

- **Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), NeurIPS** `[agarwal2021precipice]`
  - In the few-run regime, point estimates of aggregate RL performance are unreliable. The authors recommend stratified bootstrap confidence intervals, performance profiles and the interquartile mean (IQM).
  - **Bearing:** this is our primary analysis protocol for all hypotheses. IQM over (task, seed) cells with bootstrap CIs, plus performance profiles per conversion method, makes H1-H7 comparisons robust to outlier tasks.

- **Henderson, Islam, Bachman et al. (2018), AAAI** `[henderson2018matters]`
  - Deep-RL results vary greatly with random seeds, hyperparameters, environment stochasticity and implementation details. The authors propose reporting and significance-testing guidelines.
  - **Bearing:** this justifies multiple seeds per (model, method, task) cell, fixed hyperparameter budgets across rungs, and pre-registration. It is particularly relevant to the LoRA and GRPO conversions (H3).

- **Towers, Kwiatkowski, Terry et al. (2025), NeurIPS Datasets and Benchmarks** `[towers2025gymnasium]`
  - Gymnasium is the maintained standard API for RL environments (successor to OpenAI Gym), including toy-text environments such as Blackjack and FrozenLake, with explicit seeding semantics.
  - **Bearing:** this is the recommended interface for our procedurally generated tasks (reproducible seeding, standard wrappers). Its Blackjack environment is a ready reference for our risk task's rules.

- **Sutton & Barto (2018), MIT Press (2nd ed.)** `[sutton2018rl]`
  - This is the standard RL text covering MDPs, dynamic programming, Monte Carlo methods (including the classic blackjack example) and bandit algorithms (epsilon-greedy, UCB, gradient bandits).
  - **Bearing:** it provides the oracle constructions (value iteration for gridworld, optimal blackjack policy under fixed rules) and the random/oracle normalisation used for all hypotheses.

- **Gao, Tow, Abbasi et al. (2023), Zenodo (lm-evaluation-harness v0.4.0)** `[gao2023lmeval]`
  - This is the de facto open implementation of log-likelihood multiple-choice scoring (with raw and length-normalised accuracy) and generative evaluation.
  - **Bearing:** our prompt-score implementation should follow its conventions so H4 results are comparable to standard benchmark numbers.

- **Biderman, Schoelkopf, Sutawika et al. (2024), arXiv** `[biderman2024lessons]`
  - Lessons from three years of LM evaluation: results are sensitive to evaluation setup (prompt, normalisation, scoring vs generation), comparisons across papers are fragile, and the paper sets out best practices around lm-eval.
  - **Bearing:** this supports reporting full prompt templates, normalisation choices and scoring/generation side by side (H4), and releasing per-episode logs.

- **Hendrycks, Burns, Basart et al. (2021), ICLR** `[hendrycks2021mmlu]`
  - MMLU is a 57-subject multiple-choice benchmark; when introduced, most models were near chance and the largest GPT-3 ~20 points above it.
  - **Bearing:** MMLU (or a similar general-capability score) is an alternative x-axis to parameters for H1. Regressing decision quality on general capability tests whether decision-making scales beyond what general knowledge predicts.

- **Guo, Pleiss, Sun & Weinberger (2017), ICML** `[guo2017calibration]`
  - Modern deep networks are poorly calibrated, and single-parameter temperature scaling is a surprisingly effective fix.
  - **Bearing:** we should report calibration (ECE) of scored and probed policies and consider temperature-scaled action distributions (H4). This matters especially for stochastic policies in bandits (H6).

### 2.8 Cognitive and behavioural decision studies

- **Binz & Schulz (2023), PNAS** `[binz2023cognitive]`
  - GPT-3 was tested with cognitive-psychology tasks. It did well on several decision tasks, outperforming humans on a two-armed bandit, but showed *no directed exploration* and poor causal reasoning, and small vignette changes threw it off.
  - **Bearing:** this is early behavioural evidence for H6 (exploration is the weak point). It provides a template for analysing *how* models deviate (exploration style, risk attitude), not only how much.

- **Binz, Akata, Bethge et al. (2025), Nature** `[binz2025centaur]`
  - Centaur fine-tunes Llama 3.1 70B with low-rank adapters on Psych-101: trial-level data from 60,000+ participants and 10M+ choices across 160 experiments.
  - It predicts held-out human behaviour better than domain cognitive models and generalises to new cover stories, structural task modifications and new domains.
  - **Bearing:** LoRA fine-tuning on choice data generalised to *some* OOD shifts (cover stories, structure), which argues for carefully typed OOD splits in H3. It also highlights that "human-like" and "optimal" policies differ; our oracle is normative.

- **Fan, Chen, Jin & He (2024), AAAI** `[fan2024rational]`
  - Using the dictator game, Rock-Paper-Scissors and a ring-network game, the authors tested preference formation, belief updating and action choice. Even GPT-4 fell well short of human-like rationality, failing to infer beliefs from simple patterns and abandoning formed beliefs.
  - **Bearing:** this is relevant to adversarial tic-tac-toe, where exploiting an imperfect opponent requires belief formation. It shows such failures persist at frontier scale (H1 caveat).

- **Jia, Yuan, Pan, McNamara & Chen (2024), NeurIPS** `[jia2024decision]`
  - A behavioural-economics framework estimates LLM risk preference, probability weighting and loss aversion via multiple-choice-list experiments.
  - LLMs show human-like risk and loss aversion and overweight small probabilities, to varying degrees across models, and behaviour shifts with socio-demographic persona prompts.
  - **Bearing:** this is the key reference for our blackjack/risk task. Deviations from the EV-optimal policy may reflect systematic, human-like risk attitudes rather than noise, so error *direction* (too conservative vs too risky) should be analysed per rung.

- **Liu, Geng, Peterson, Sucholutsky & Griffiths (2025), ICLR** `[liu2025rational]`
  - When predicting or interpreting human choices, LLMs (GPT-4o/4-Turbo, Llama-3-8B/70B, Claude 3 Opus) assume people are more rational than they are, aligning with expected-value theory.
  - **Bearing:** this suggests LLMs carry an internal EV-maximising "theory of choice". If they apply it to their own decisions, larger models should approach EV-optimal play in risk tasks (H1), possibly more so with explicit reasoning (H7).

### 2.9 Open-weight model families

Per-size details, repository ids and licences are in `docs/model_ladders.md`; this section notes what each report means for the design.

- **Qwen Team: Yang, Yang et al. (2024), arXiv** `[qwen2024qwen25]`
  - Qwen2.5 provides base and instruct models at 0.5B, 1.5B, 3B, 7B, 14B, 32B and 72B. Pre-training data was scaled to 18T tokens, and post-training used >1M SFT samples plus multi-stage RL (DPO and GRPO).
  - **Bearing:** this is the cleanest seven-rung base+instruct ladder (H1, H2) and the family most used in recent LLM-agent RL papers, which eases comparison. Licences differ at 3B and 72B.

- **Yang, Li, Yang, Zhang et al. (2025), arXiv** `[yang2025qwen3]`
  - Qwen3 has dense models from 0.6B to 32B and MoE models (30B-A3B, 235B-A22B), unifying thinking and non-thinking modes in one checkpoint with a user-set thinking budget.
  - **Bearing:** this enables a within-checkpoint causal test of H7 (thinking on vs off vs budget) across six dense sizes. Base checkpoints exist only up to 14B (plus 30B-A3B), which limits H2 at the top of the ladder.

- **Qwen Team (2025), GitHub** `[qwen2025qwen3github]`
  - The repository documents the Qwen3 release (29 Apr 2025) and the 2507 refresh, which split thinking and non-thinking into separate checkpoints (Instruct-2507 / Thinking-2507) at 4B, 30B-A3B and 235B-A22B with 256K context. It also documents the `enable_thinking` and `/think` `/no_think` controls.
  - **Bearing:** this provides a second H7 contrast, separately trained thinking vs non-thinking models at fixed size.

- **Qwen Team (2026), GitHub** `[qwen2026github]`
  - The repository records the 2026 ladders:
    - Qwen3.5 (16 Feb - 2 Mar 2026): 0.8B, 2B, 4B, 9B, 27B dense plus 35B-A3B, 122B-A10B and 397B-A17B MoE;
    - Qwen3.6 (Apr 2026): 27B and 35B-A3B;
    - Qwen3.8 (Aug 2026): 27B and 2.4T-A95B.
  - It also records Qwen3-Next-80B-A3B (Sep 2025) and graded `reasoning_effort` controls.
  - **Bearing:** this is the most recent same-family ladder (useful for replication) and its graded reasoning effort suits H7 budget sweeps. However, it has a vision encoder throughout and little literature yet.

- **Ben Allal, Lozhkov, Bakouch et al. (2025), arXiv** `[allal2025smollm2]`
  - SmolLM2 provides 135M, 360M and 1.7B models (base and instruct). The 1.7B was trained on ~11T tokens with multi-stage data mixing and new datasets (FineMath, Stack-Edu, SmolTalk), and outperformed Qwen2.5-1.5B and Llama-3.2-1B.
  - **Bearing:** this supplies sub-1B rungs needed for H5's "costly below 1B" and the extreme low end of H1, with heavy over-training relevant to (Kumar et al., 2025).

- **Hugging Face (2025), blog** `[hf2025smollm3]`
  - SmolLM3 is a fully open 3B model trained on ~11T tokens, with dual-mode think/no_think reasoning, six languages and 128K context via YaRN (trained at 64K).
  - **Bearing:** this is a small model with a thinking toggle, adding an independent family to Qwen3 for H7 at 3B.

- **Gemma Team (2025), arXiv** `[gemma2025gemma3]`
  - Gemma 3 is a distilled family (1B, 4B, 12B, 27B; 270M added later) with pre-trained and instruction-tuned checkpoints. It has 128K context (32K at 1B), an increased ratio of local to global attention, and vision for 4B and above.
  - The authors report that Gemma3-4B-IT is competitive with Gemma2-27B-IT.
  - **Bearing:** this is a pt+it ladder for H1/H2. Its very large 256k-entry vocabulary makes embeddings a large fraction of small rungs, so a non-embedding x-axis matters. Licence is the Gemma Terms of Use.

- **Google (2025), blog** `[google2025gemma3n]`
  - Gemma 3n E2B and E4B are "effective-parameter" models (raw ~5B and ~8B) built with MatFormer and per-layer embeddings that can be cached off-accelerator. They handle text, image and audio.
  - **Bearing:** their effective vs raw parameter counts do not fit a single scale axis, so they are better excluded from H1 fits or analysed separately.

- **Google (2026), blog** `[google2026gemma4]`
  - Gemma 4 (April 2026) provides E2B, E4B, a 26B MoE (A4B) and a 31B dense model, with configurable thinking, multimodal input and (per contemporary coverage) an Apache-2.0 licence, a change from the Gemma terms.
  - **Bearing:** this is a permissively licensed Google ladder with a thinking switch (H7), but mixed effective/MoE/dense rungs complicate H1.

- **Grattafiori, Dubey et al. (2024), arXiv** `[grattafiori2024llama3]`
  - The Llama 3 herd report covers dense 8B, 70B and 405B base and instruct models with 128K context.
  - **Bearing:** Llama 3.x sizes come from *different* releases (3.1: 8B/70B/405B; 3.2: 1B/3B; 3.3: 70B instruct only; see (Meta, 2025)), so Llama is not a single-recipe ladder. Use it as a cross-family check rather than a primary H1 ladder.

- **Meta (2025), GitHub** `[meta2025llamamodels]`
  - The release table gives dates and context lengths for Llama 3.1 (Jul 2024), 3.2 (Sep 2024), 3.3 (Dec 2024) and Llama 4 (Apr 2025). Llama 4 comprises Scout-17B-16E (10M context) and Maverick-17B-128E (1M context) MoE models.
  - **Bearing:** the MoE models with 17B active parameters cannot be placed on a dense-parameter axis without choosing active vs total parameters, so they are out of scope for H1 fits.

- **Biderman, Schoelkopf, Anthony et al. (2023), ICML** `[biderman2023pythia]`
  - Pythia is 16 models (8 sizes, 70M-12B, each with a deduplicated variant) trained on public data in exactly the same order, with 154 checkpoints each.
  - **Bearing:** this is the gold-standard scale control (identical data and order), and it also allows training-time vs scale analyses. It is base-only and weak in absolute terms, so it suits H1 (low end), probes and LoRA but not H2.

- **Team OLMo, Walsh, Soldaini et al. (2025), arXiv** `[olmo2025olmo2]`
  - OLMo 2 is a fully open family (7B, 13B, 32B in the paper, plus a later 1B) with released data, code and checkpoints, and base plus instruct versions.
  - **Bearing:** this is a four-rung fully open base+instruct ladder (H1, H2) with data available for contamination checks.

- **Team Olmo, Ettinger, Bertsch et al. (2025), arXiv** `[olmo2025olmo3]`
  - Olmo 3 provides fully open 7B and 32B models (Base, Instruct, Think, RL-Zero), pre-trained on Dolma 3 (~5.9T tokens) with 65K context, releasing the full "model flow".
  - **Bearing:** Think vs Instruct from the same base gives a clean H7 contrast with open data, but only two sizes.

- **OpenAI (2025), arXiv** `[openai2025gptoss]`
  - The gpt-oss-120b (116.8B total / 5.1B active) and gpt-oss-20b (20.9B / 3.6B) MoE reasoning models were post-trained with MoE weights in MXFP4 (~4.25 bits/parameter) and offer low/medium/high reasoning effort under Apache 2.0.
  - **Bearing:** graded reasoning effort suits H7 budget sweeps. Native 4-bit MoE weights are a natural case for H5, but there is no base model and no small rungs.

- **Abdin, Aneja, Behl et al. (2024), arXiv** `[abdin2024phi4]`
  - Phi-4 is a 14B model trained with a synthetic-data-centric curriculum; it surpasses its GPT-4 teacher on STEM-focused QA.
  - **Bearing:** it is a recipe outlier (synthetic data) useful as an off-ladder comparison point. The Phi-4 family (mini 3.8B, 14B) is not a matched ladder.

- **Abdin, Agarwal, Awadallah et al. (2025), arXiv** `[abdin2025phi4reasoning]`
  - Phi-4-reasoning is a 14B model fine-tuned from Phi-4 on curated prompts with o3-mini reasoning traces. The "plus" variant adds outcome-based RL. Both outperform DeepSeek-R1-Distill-Llama-70B.
  - **Bearing:** this is a same-base reasoning vs non-reasoning contrast at 14B for H7.

- **Mistral AI (2025), announcement** `[mistral2025small3]`
  - Mistral Small 3 is a latency-optimised 24B model released as pre-trained and instruction-tuned checkpoints under Apache 2.0, which Mistral reports as on par with Llama 3.3 70B instruct at >3x the speed.
  - **Bearing:** this is a single-size base+instruct pair for H2 at 24B and a latency reference for H7's efficiency metric.

- **Mistral AI (2025), announcement** `[mistral2025small31]`
  - Mistral Small 3.1 (24B) adds multimodal input and 128K context, again with a base checkpoint and Apache 2.0.
  - **Bearing:** this gives a long-context option for history-heavy bandit prompts at 24B.

- **Mistral AI (2025), announcement** `[mistral2025mistral3]`
  - Mistral 3 (December 2025) introduced the Ministral 3 dense models at 3B, 8B and 14B, each released as base, instruct and reasoning variants with image understanding, under Apache 2.0.
  - **Bearing:** a three-rung ladder where every rung has base/instruct/reasoning triplets is ideal for H2 and H7 at small scale.

---

## 3. Gaps our study fills

The following assessment tries to be honest: several hypotheses already have substantial partial answers.

1. **No controlled ladder × conversion-method × task-structure study exists.**
   - Agent benchmarks mostly evaluate frontier or heterogeneous model sets (Paglieri et al., 2025; Ruoss et al., 2025; Liu et al., 2024; Duan et al., 2024).
   - Scale-resolved studies use one family and two or three sizes (Schmied et al., 2026; Jiwatode et al., 2026) or one task type (Monea et al., 2025).
   - Conversion-method papers use one method at one or two sizes (Carta et al., 2023; Tan et al., 2024; Chu et al., 2025; Dilkes et al., 2025).
   - DeadEye's crossed design (≥3 same-recipe ladders × 4-5 conversion methods × 6 task structures) and its random-to-oracle normalisation with IQM and bootstrap CIs (Agarwal et al., 2021) have no direct precedent.
   - *Partial prior answers:* GLAM and TWOSOME already show scoring-based policies work; Chu et al. already show SFT-vs-RL generalisation differences.

2. **H1 (log-linear, task-dependent slope) is plausible but untested in this form.**
   - Supporting evidence: long-horizon execution scales with size without diminishing returns (Sinha et al., 2026), frequency bias fades from 2B to 27B (Schmied et al., 2026), and multi-size bandit ICL (Monea et al., 2025).
   - Nobody has fit slopes per task structure with a principled x-axis. That axis should use non-embedding parameters (Kaplan et al., 2020), handle MoE active vs total, and control for tokens-per-parameter (Hoffmann et al., 2022).
   - Metrics must be continuous to avoid mirage effects (Schaeffer et al., 2023).
   - *Honest caveat:* our tasks are small and synthetic, so slopes may not transfer to open-ended agentic work.

3. **H2 (instruct > base under prompting, gap vanishes under probe/LoRA) has strong priors in both directions.**
   - LIMA and URIAL suggest alignment is mostly style (Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, Lewis, Zettlemoyer and Levy, 2023; Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula and Choi, 2024). That predicts the second clause but also threatens the first (base + few-shot may suffice).
   - First-token/text mismatches make scoring-based comparisons of instruct models tricky (Wang, Ma, Hu, Weber-Genzel, Röttger, Kreuter, Hovy and Plank, 2024; Wang, Hu, Ma, Röttger and Plank, 2024).
   - No one has tested base vs instruct under probe and LoRA on decision tasks across a ladder, and Qwen3's missing large base checkpoints constrain the design.

4. **H3 (fine-tuned small ≈ 10x larger zero-shot in-distribution, not OOD) is half-answered.**
   - The in-distribution half is supported for bandits (Nie et al., 2025) and Frozen Lake (3B > 72B after RL) (Dilkes et al., 2025).
   - The OOD half is supported for SFT (Chu et al., 2025; Ross et al., 2011) but contradicted for diverse-task fine-tuning (Tajwar et al., 2025), LLM-initialised RL policies (Szot et al., 2024) and cognitive-task fine-tuning (Binz et al., 2025).
   - DeadEye's contribution is *typed* OOD splits that procedural generation makes possible: parameter shift (grid size, payoff scale), surface shift (renamed features or actions), opponent shift. These are run at matched 10x size ratios across families.
   - *Design note:* for bandits the clairvoyant oracle never explores, so behaviour cloning on oracle trajectories cannot teach exploration (Laskin et al., 2023). The SFT teacher should be UCB/Thompson (Auer et al., 2002), while normalisation still uses the oracle.

5. **H4 (logit scoring removes format failures) is well motivated but unmeasured for sequential decisions.**
   - Prior work is on QA multiple choice (Holtzman et al., 2021; Robinson and Wingate, 2023; Zheng et al., 2024; Sclar et al., 2024) or uses scoring without comparing to generation across scale (Carta et al., 2023; Tan et al., 2024).
   - DeadEye can quantify format-failure rates against scale and test whether scoring closes the gap. It should also check whether scoring introduces its own biases (option-ID priors, action-length effects, surface-form competition).
   - *Expected nuance:* scoring helps most for base and small models (Wang, Hu, Ma, Röttger and Plank, 2024).

6. **H5 (4-bit quantisation free above ~3B, costly below 1B) has a mechanism but no decision-task evidence.**
   - Quantisation studies measure perplexity, MMLU or maths (Dettmers and Zettlemoyer, 2023; Zheng et al., 2026; Liu, Sun, Zhang, Bai, Yu, Yu, Yuan and Hou, 2025).
   - Precision scaling laws predict over-trained small models suffer most (Kumar et al., 2025).
   - Evidence on multi-step decision quality vs scale under 4-bit (GPTQ/AWQ/NF4), including the interaction with thinking mode, is missing.

7. **H6 (exploration poor without scaffolding) is largely established; DeadEye's value is resolution and robustness.**
   - (Krishnamurthy et al., 2024), (Schmied et al., 2026), (Harris and Slivkins, 2026) and (Binz and Schulz, 2023) already show greedy or under-exploring LLMs, and (Nie et al., 2025) shows scaffolds and distillation help.
   - What is new is the full scale curve across several families, whether probes or LoRA change it, and separating exploration from exploitation failures (Harris and Slivkins, 2026) and from option-ID or recency artefacts (Zheng et al., 2024; Zhao et al., 2021).
   - *Honest framing:* treat H6 as a replication and extension, not a discovery.

8. **H7 (thinking helps planning, can hurt small models, hurts efficiency) has fragmented support.**
   - The claim that reasoning can hurt and is complexity-dependent is supported by (Liu, Geng, Wu, Sucholutsky, Lombrozo and Griffiths, 2025), (Shojaee et al., 2025), (Gema et al., 2025), (Cuadron et al., 2025) and (Li et al., 2025). The closest scale × thinking × game study is (Jiwatode et al., 2026).
   - Thinking toggles inside the same weights (Qwen3, SmolLM3, Gemma 4, gpt-oss effort levels, Olmo 3 Think vs Instruct) make a cleaner causal test possible than earlier cross-model comparisons.
   - The *latency-normalised* efficiency analysis across ≥5 sizes and several task structures is new.

9. **Methodological gaps we address.**
   - RL-grade statistics (IQM, stratified bootstrap, performance profiles) are rare in LLM-agent papers.
   - Prompt-format ranges (Sclar et al., 2024) are rarely reported.
   - Parse/format failures are rarely separated from wrong-but-legal decisions (Huang et al., 2022).
   - Contamination of classic games is rarely addressed. Blackjack basic strategy and tic-tac-toe are heavily documented online, so procedural rule variants (non-standard payouts, board sizes) should test reasoning rather than recall.

---

## 4. Unverified candidates

*Note added after the bibliography audit: `docs/references_audit.md` records the final verification status of every entry in `paper/refs.bib`; sections 4 and 5 describe the state at the time of the review.*

These were found or suggested but could not be fully verified. They are **not** in `paper/refs.bib` unless noted.

### Not in the bib

| Candidate | What could not be confirmed |
|---|---|
| "On Emotion-Sensitive Decision Making of Small Language Model Agents", arXiv:2604.06562 (2026) | Author list not confirmed. Abstract (Qwen3 small agents on game-theoretic tasks; thinking mode can *amplify* sensitivity to induced emotion) is relevant to H7. |
| "TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices", arXiv:2511.22138 (2025) | Authors not confirmed. Reported numbers: on BFCL, xLAM-2-3B scored 65.7% overall but Qwen3-1.7B only 16.9% on multi-turn. |
| "Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs", arXiv:2510.03847 (2025) | Authors not confirmed. |
| Llama 4 technical report | A catalog page states the companion arXiv preprint was withdrawn or redacted. Llama 4 is cited via the official `meta-llama/llama-models` repository instead. |
| Gemma 4 technical report | The `google-deepmind/gemma` README links Gemma 1-4 reports, but the Gemma 4 report title and identifier were not retrieved. Cited via Google's announcement blog. |
| Qwen3.5 / Qwen3.6 / Qwen3.8 technical reports | Not found. Cited via the official `QwenLM/Qwen3.8` GitHub README. |
| Gemma 3 270M announcement | Details (14 Aug 2025; ~170M embedding + ~100M transformer parameters) come from third-party coverage only. Recorded in `model_ladders.md` and flagged there. |

### In the bib, with caveats

- **Venues not confirmed, so cited as arXiv:**
  - SmolLM2 (possibly COLM 2025);
  - "Look at the Text" (possibly COLM 2024);
  - PAPRIKA / "Training a Generally Curious Agent";
  - "Quantization Hurts Reasoning?" (possibly COLM 2025);
  - RAGEN;
  - "The Danger of Overthinking";
  - Alain & Bengio (indexed under ICLR 2017, probably the workshop track);
  - GameBench (also a NeurIPS 2024 workshop).
- **Author lists only partly confirmed (bib ends with `and others`):** Kadavath et al.; Skean et al.; Orgad et al.; Chu et al.; LIMA; RAGEN; Cuadron et al.; Gema et al.; Li et al.; Wang et al. ("My Answer is C"); Biderman et al. (Lessons); Qwen2.5/Qwen3 reports; Llama 3 herd; SmolLM2; OLMo 2/3; Phi-4-reasoning; Centaur.
- **DeepSeek-R1 Nature version:** title and DOI confirmed (10.1038/s41586-025-09422-z); volume and pages not confirmed. Recorded in the note field of `deepseekai2025r1`.
- **Schmied et al.:** also appears on an ICML 2025 virtual page, probably a workshop. Cited at ICLR 2026 per the ICLR proceedings listing.

### Corrections to the original candidate list (all reflected in the bib)

- **Krishnamurthy et al. 2024:** the fourth author is **Cyril Zhang**, not Schapire. The full list is Krishnamurthy, Harris, Foster, Zhang, Slivkins.
- **EVOLvE:** the full title is "EVOLvE: Evaluating and Optimizing LLMs For **In-Context** Exploration". Published at **ICML 2025** (PMLR 267).
- **Chinchilla:** the NeurIPS 2022 proceedings title is "An Empirical Analysis of Compute-Optimal Large Language Model Training". The arXiv title is "Training Compute-Optimal Large Language Models".
- **GPTQ:** published at ICLR 2023 under the name **OPTQ**. The arXiv title retains "GPTQ".
- **AWQ:** published at **MLSys 2024** with "On-Device" in the title. The arXiv title omits it.
- **GTBench:** published at **NeurIPS 2024** with "Strategic Reasoning **Capabilities**" in the title. arXiv says "Limitations".
- **Centaur:** published in **Nature 644 (2025)** as "A foundation model to predict and capture human cognition".
- **DeepSeek-R1:** also published in **Nature (2025)** under a different title.
- **Gymnasium:** now **NeurIPS 2025 Datasets and Benchmarks**.
- **TextArena:** authors are Guertler, Cheng, Yu, Liu, Choshen, Tan.
- **Llama 3 herd:** first author is Grattafiori in arXiv v3 (Dubey in v2).
- **Let Me Speak Freely?:** published at **EMNLP 2024 Industry Track**. The Anthology subtitle differs from arXiv.
- **Mind Your Step:** published at **ICML 2025**.
- **The Illusion of Thinking:** published at **NeurIPS 2025**.
- **Harris & Slivkins:** published at **UAI 2026**.
- **Reflexion:** the arXiv v4 author list includes Edward Berman.
- **DPO:** NeurIPS proceedings list Manning before Ermon.
- **Belcak et al.:** correct as given (arXiv 2506.02153).
- **Binz & Schulz:** PNAS 120(6), e2218523120.
- **OLMo 2:** the paper covers 7B/13B/32B. The 1B (0425) came later.
- **Qwen3:** base checkpoints were **not** released for 32B or 235B-A22B.

---

## 5. Recent work to watch (2025-2026)

**Directly on our hypotheses**

- **(Schmied et al., 2026)** (ICLR 2026): greediness, frequency bias and the knowing-doing gap in 2B-27B models; RL fine-tuning on chain-of-thought fixes part of it. *H1, H6, H7, GRPO stretch.*
- **(Jiwatode et al., 2026)** (CoG 2026): Qwen3 scale × thinking mode × planning horizon on spatial games. *H1, H7; closest design to ours.*
- **(Sinha et al., 2026)** (ICLR 2026): execution horizon keeps scaling with model size. *H1 horizon-dependent slopes.*
- **(Harris and Slivkins, 2026)** (UAI 2026): exploitation, not only exploration, fails; reasoning models help but are slow. *H6, H7.*
- **(Dilkes et al., 2025)**: MS-GRPO makes a 3B model beat a 72B one on Frozen Lake. *H3 in-distribution.*
- **(Liu et al., 2026)** (ICLR 2026) and **(Wang et al., 2025)**: multi-turn RL for small LLM agents on games and bandits; stability pitfalls. *GRPO stretch, H6.*
- **(Tajwar et al., 2025)**: diverse-task fine-tuning transfers exploration to unseen tasks. *H3 OOD counter-evidence.*
- **(Monea et al., 2025)** (COLM 2025): in-context bandit RL from 500M to 70B. *H1, H6.*
- **(Chu et al., 2025)** (ICML 2025): SFT memorises, RL generalises. *H3.*
- **(Kumar et al., 2025)** (ICLR 2025), **(Zheng et al., 2026)**, **(Liu, Sun, Zhang, Bai, Yu, Yu, Yuan and Hou, 2025)**: quantisation hurts over-trained and small models, and reasoning, more. *H5, H5 × H7.*
- **(Li et al., 2025)**, **(Gema et al., 2025)**, **(Shojaee et al., 2025)**, **(Liu, Geng, Wu, Sucholutsky, Lombrozo and Griffiths, 2025)**, **(Cuadron et al., 2025)**: when thinking hurts. *H7.*
- **(Snell et al., 2025)**: test-time compute vs parameters. *H7 efficiency framing.*
- **(Belcak et al., 2025)**: the SLM-for-agents position paper. *Motivation.*

**Model releases since mid-2025 relevant to ladder choice** (details in `model_ladders.md`)

- Qwen3-2507 thinking/instruct split (Jul-Aug 2025).
- Qwen3-Next-80B-A3B (Sep 2025).
- Qwen3.5 0.8B-397B (Feb-Mar 2026), Qwen3.6 (Apr 2026), Qwen3.8 (Aug 2026).
- Gemma 3 270M (Aug 2025), Gemma 3n (2025), Gemma 4 (Apr 2026, Apache 2.0).
- SmolLM3 (Jul 2025).
- Olmo 3 and Olmo 3.1 (Nov-Dec 2025).
- Ministral 3 at 3B/8B/14B with base/instruct/reasoning variants (Dec 2025).
- gpt-oss (Aug 2025).

**Unverified but worth tracking (Section 4):** the emotion-sensitive SLM-agent study (arXiv:2604.06562) and the TinyLLM edge-agent benchmark (arXiv:2511.22138).

## Reference list

*Generated mechanically from `paper/refs.bib` after the 2026-10-08 audit (see `docs/references_audit.md`). In-text citations above follow the labels that `agsm.bst` with natbib produces for the paper.*

Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., Harrison, M., Hewett, R. J., Javaheripi, M., Kauffmann, P., Lee, J. R., Lee, Y. T., Li, Y., Liu, W., Mendes, C. C. T., Nguyen, A., Price, E., de Rosa, G., Saarikivi, O., Salim, A., Shah, S., Wang, X., Ward, R., Wu, Y., Yu, D., Zhang, C. and Zhang, Y. (2024) 'Phi-4 technical report', *arXiv preprint* arXiv:2412.08905. Available at: <https://arxiv.org/abs/2412.08905>.

Abdin, M., Agarwal, S., Awadallah, A., Balachandran, V., Behl, H., Chen, L., de Rosa, G., Gunasekar, S., Javaheripi, M., Joshi, N., Kauffmann, P., Lara, Y., Mendes, C. C. T., Mitra, A., Nushi, B., Papailiopoulos, D., Saarikivi, O., Shah, S., Shrivastava, V., Vineet, V., Wu, Y., Yousefi, S. and Zheng, G. (2025) 'Phi-4-reasoning technical report', *arXiv preprint* arXiv:2504.21318. Available at: <https://arxiv.org/abs/2504.21318>.

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. and Bellemare, M. G. (2021) 'Deep reinforcement learning at the edge of the statistical precipice', in *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. Curran Associates, Inc. Also available as arXiv:[2108.13264](https://arxiv.org/abs/2108.13264).

Alain, G. and Bengio, Y. (2017) 'Understanding intermediate layers using linear classifier probes', in *5th International Conference on Learning Representations (ICLR 2017), Workshop Track*. OpenReview.net. Also available as arXiv:[1610.01644](https://arxiv.org/abs/1610.01644).

Auer, P., Cesa-Bianchi, N. and Fischer, P. (2002) 'Finite-time analysis of the multiarmed bandit problem', *Machine Learning*, 47, pp. 235–256. doi: [10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352).

Bakouch, E., Ben Allal, L., Lozhkov, A., Tazi, N., Tunstall, L., Patiño, C. M., Beeching, E., Roucher, A., Reedi, A. J., Gallouédec, Q., Rasul, K., Habib, N., Fourrier, C., Kydlíček, H., Penedo, G., Larcher, H., Morlon, M., Srivastav, V., Lochner, J., Nguyen, X.-S., Raffel, C., von Werra, L. and Wolf, T. (2025) 'SmolLM3: smol, multilingual, long-context reasoner', Hugging Face blog (official release write-up of SmolLM3). Available at: <https://huggingface.co/blog/smollm3> (Accessed: 8 October 2026).

Belcak, P., Heinrich, G., Diao, S., Fu, Y., Dong, X., Muralidharan, S., Lin, Y. C. and Molchanov, P. (2025) 'Small language models are the future of agentic AI', *arXiv preprint* arXiv:2506.02153. Available at: <https://arxiv.org/abs/2506.02153>.

Belinkov, Y. (2022) 'Probing classifiers: Promises, shortcomings, and advances', *Computational Linguistics*, 48(1), pp. 207–219. doi: [10.1162/coli\_a\_00422](https://doi.org/10.1162/coli_a_00422).

Ben Allal, L., Lozhkov, A., Bakouch, E., Martín Blázquez, G., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Piqueres Lajarín, A., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., Raffel, C., von Werra, L. and Wolf, T. (2025) 'SmolLM2: When smol goes big – data-centric training of a small language model', *arXiv preprint* arXiv:2502.02737. Available at: <https://arxiv.org/abs/2502.02737>.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L. and van der Wal, O. (2023) 'Pythia: A suite for analyzing large language models across training and scaling', in *Proceedings of the 40th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 202, pp. 2397–2430. PMLR. Also available as arXiv:[2304.01373](https://arxiv.org/abs/2304.01373).

Biderman, S., Schoelkopf, H., Sutawika, L., Gao, L., Tow, J., Abbasi, B., Aji, A. F., Ammanamanchi, P. S., Black, S., Clive, J., DiPofi, A., Etxaniz, J., Fattori, B., Forde, J. Z., Foster, C., Hsu, J., Jaiswal, M., Lee, W. Y., Li, H., Lovering, C., Muennighoff, N., Pavlick, E., Phang, J., Skowron, A., Tan, S., Tang, X., Wang, K. A., Winata, G. I., Yvon, F. and Zou, A. (2024) 'Lessons from the trenches on reproducible evaluation of language models', *arXiv preprint* arXiv:2405.14782. Available at: <https://arxiv.org/abs/2405.14782>.

Binz, M. and Schulz, E. (2023) 'Using cognitive psychology to understand GPT-3', *Proceedings of the National Academy of Sciences*, 120(6), e2218523120. doi: [10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120).

Binz, M., Akata, E., Bethge, M. et al. (2025) 'A foundation model to predict and capture human cognition', *Nature*, 644(8078), pp. 1002–1009. Preprint version: arXiv:2410.20268, “Centaur: a foundation model of human cognition”. doi: [10.1038/s41586-025-09215-4](https://doi.org/10.1038/s41586-025-09215-4).

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I. and Amodei, D. (2020) 'Language models are few-shot learners', in *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. Curran Associates, Inc. Also available as arXiv:[2005.14165](https://arxiv.org/abs/2005.14165).

Carta, T., Romac, C., Wolf, T., Lamprier, S., Sigaud, O. and Oudeyer, P.-Y. (2023) 'Grounding large language models in interactive environments with online reinforcement learning', in *Proceedings of the 40th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 202, pp. 3676–3713. PMLR. Also available as arXiv:[2302.02662](https://arxiv.org/abs/2302.02662).

Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A. and Mordatch, I. (2021) 'Decision Transformer: Reinforcement learning via sequence modeling', in *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. Curran Associates, Inc. Also available as arXiv:[2106.01345](https://arxiv.org/abs/2106.01345).

Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S. and Ma, Y. (2025) 'SFT memorizes, RL generalizes: A comparative study of foundation model post-training', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 10818–10838. PMLR. Also available as arXiv:[2501.17161](https://arxiv.org/abs/2501.17161).

Costarelli, A., Allen, M., Hauksson, R., Sodunke, G., Hariharan, S., Cheng, C., Li, W., Clymer, J. and Yadav, A. (2024) 'GameBench: Evaluating strategic reasoning abilities of LLM agents', in *Language Gamification Workshop at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)*. Non-archival workshop paper; also available as arXiv:2406.06613.

Cuadron, A., Li, D., Ma, W., Wang, X., Wang, Y., Zhuang, S., Liu, S., Schroeder, L. G., Xia, T., Mao, H., Thumiger, N., Desai, A., Stoica, I., Klimovic, A., Neubig, G. and Gonzalez, J. E. (2025) 'The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks', *arXiv preprint* arXiv:2502.08235. Available at: <https://arxiv.org/abs/2502.08235>.

Dettmers, T. and Zettlemoyer, L. (2023) 'The case for 4-bit precision: k-bit inference scaling laws', in *Proceedings of the 40th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 202, pp. 7750–7774. PMLR. Also available as arXiv:[2212.09720](https://arxiv.org/abs/2212.09720).

Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023) 'QLoRA: Efficient finetuning of quantized LLMs', in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Curran Associates, Inc. Also available as arXiv:[2305.14314](https://arxiv.org/abs/2305.14314).

Dilkes, J., Yazdanpanah, V. and Stein, S. (2025) 'Reinforced language models for sequential decision making', *arXiv preprint* arXiv:2508.10839. Available at: <https://arxiv.org/abs/2508.10839>.

Duan, J., Zhang, R., Diffenderfer, J., Kailkhura, B., Sun, L., Stengel-Eskin, E., Bansal, M., Chen, T. and Xu, K. (2024) 'GTBench: Uncovering the strategic reasoning capabilities of LLMs via game-theoretic evaluations', in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Curran Associates, Inc. Title as in the proceedings index; the paper PDF and arXiv:2402.12348 read “Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations”.

Fan, C., Chen, J., Jin, Y. and He, H. (2024) 'Can large language models serve as rational players in game theory? A systematic analysis', *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(16), pp. 17960–17967. doi: [10.1609/aaai.v38i16.29751](https://doi.org/10.1609/aaai.v38i16.29751).

Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2023) 'OPTQ: Accurate quantization for generative pre-trained transformers', in *The Eleventh International Conference on Learning Representations (ICLR 2023)*. OpenReview.net. Widely cited as GPTQ; preprint version: arXiv:2210.17323, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K. and Zou, A. (2023) 'A framework for few-shot language model evaluation', Zenodo software release of EleutherAI's lm-evaluation-harness, version v0.4.0, 4 December 2023. doi: [10.5281/zenodo.10256836](https://doi.org/10.5281/zenodo.10256836). (Accessed: 8 October 2026).

Gema, A. P., Hägele, A., Chen, R., Arditi, A., Goldman-Wetzler, J., Fraser-Taliente, K., Sleight, H., Petrini, L., Michael, J., Alex, B., Minervini, P., Chen, Y., Benton, J. and Perez, E. (2025) 'Inverse scaling in test-time compute', *Transactions on Machine Learning Research*. Available at: <https://openreview.net/forum?id=NXgyHW1c7M>.

Gemma Team (2025) 'Gemma 3 technical report', *arXiv preprint* arXiv:2503.19786. Google DeepMind. Available at: <https://arxiv.org/abs/2503.19786>.

Google (2025) 'Introducing Gemma 3n: The developer guide', Google Developers Blog, 26 June 2025. Available at: <https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/> (Accessed: 8 October 2026).

Google (2026) 'Gemma 4: Byte for byte, the most capable open models', Google blog, 2 April 2026 (announcement by C. Farabet and O. Lacombe, Google DeepMind). Available at: <https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/> (Accessed: 8 October 2026).

Grattafiori, A., Dubey, A. et al. (2024) 'The Llama 3 herd of models', *arXiv preprint* arXiv:2407.21783. Llama Team, AI at Meta. Available at: <https://arxiv.org/abs/2407.21783>.

Guertler, L., Cheng, B., Yu, S., Liu, B., Choshen, L. and Tan, C. (2025) 'TextArena', *arXiv preprint* arXiv:2504.11442. Available at: <https://arxiv.org/abs/2504.11442>.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017) 'On calibration of modern neural networks', in *Proceedings of the 34th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 70, pp. 1321–1330. PMLR. Also available as arXiv:[1706.04599](https://arxiv.org/abs/1706.04599).

Guo, D., Yang, D., Zhang, H., Song, J. et al. (2025) 'DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning', *Nature*, 645(8081), pp. 633–638. Preprint version: DeepSeek-AI, arXiv:2501.12948, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”. doi: [10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z).

Harris, K. and Slivkins, A. (2026) 'Should you use your large language model to explore or exploit?', in *Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence*, Proceedings of Machine Learning Research, vol. 337, pp. 2008–2058. PMLR. Also available as arXiv:[2502.00225](https://arxiv.org/abs/2502.00225).

Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X. and Sontag, D. (2023) 'TabLLM: Few-shot classification of tabular data with large language models', in *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*, Proceedings of Machine Learning Research, vol. 206, pp. 5549–5581. PMLR. Also available as arXiv:[2210.10723](https://arxiv.org/abs/2210.10723).

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018) 'Deep reinforcement learning that matters', *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1). doi: [10.1609/aaai.v32i1.11694](https://doi.org/10.1609/aaai.v32i1.11694).

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J. (2021) 'Measuring massive multitask language understanding', in *The Ninth International Conference on Learning Representations (ICLR 2021)*. OpenReview.net. Also available as arXiv:[2009.03300](https://arxiv.org/abs/2009.03300).

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. and Sifre, L. (2022) 'An empirical analysis of compute-optimal large language model training', in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. Curran Associates, Inc. Title as in the proceedings index; the paper PDF and arXiv:2203.15556 read “Training Compute-Optimal Large Language Models”.

Holtzman, A., West, P., Shwartz, V., Choi, Y. and Zettlemoyer, L. (2021) 'Surface form competition: Why the highest probability answer isn't always right', in *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 7038–7051. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic. doi: [10.18653/v1/2021.emnlp-main.564](https://doi.org/10.18653/v1/2021.emnlp-main.564).

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2022) 'LoRA: Low-rank adaptation of large language models', in *The Tenth International Conference on Learning Representations (ICLR 2022)*. OpenReview.net. Also available as arXiv:[2106.09685](https://arxiv.org/abs/2106.09685).

Huang, W., Abbeel, P., Pathak, D. and Mordatch, I. (2022) 'Language models as zero-shot planners: Extracting actionable knowledge for embodied agents', in *Proceedings of the 39th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 162, pp. 9118–9147. PMLR. Also available as arXiv:[2201.07207](https://arxiv.org/abs/2201.07207).

Jia, J., Yuan, Z., Pan, J., McNamara, P. E. and Chen, D. (2024) 'Decision-making behavior evaluation framework for LLMs under uncertain context', in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Curran Associates, Inc. Also available as arXiv:[2406.05972](https://arxiv.org/abs/2406.05972).

Jiwatode, M., Fuchs, R., Schmöcker, R., Rosenhahn, B. and Dockhorn, A. (2026) 'Spatial reasoning in LLM game agents: Impact of causal context and multi-step planning', *arXiv preprint* arXiv:2607.22732. The authors' arXiv comment states that the paper is to be published at the IEEE Conference on Games (CoG) 2026. Available at: <https://arxiv.org/abs/2607.22732>.

Kadavath, S., Conerly, T., Askell, A., Henighan, T. et al. (2022) 'Language models (mostly) know what they know', *arXiv preprint* arXiv:2207.05221. Available at: <https://arxiv.org/abs/2207.05221>.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J. and Amodei, D. (2020) 'Scaling laws for neural language models', *arXiv preprint* arXiv:2001.08361. Available at: <https://arxiv.org/abs/2001.08361>.

Krishnamurthy, A., Harris, K., Foster, D. J., Zhang, C. and Slivkins, A. (2024) 'Can large language models explore in-context?', in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Curran Associates, Inc. Also available as arXiv:[2403.15371](https://arxiv.org/abs/2403.15371).

Kumar, T., Ankner, Z., Spector, B. F., Bordelon, B., Muennighoff, N., Paul, M., Pehlevan, C., Ré, C. and Raghunathan, A. (2025) 'Scaling laws for precision', in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. OpenReview.net. Also available as arXiv:[2411.04330](https://arxiv.org/abs/2411.04330).

Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S., Steigerwald, R., Strouse, D., Hansen, S., Filos, A., Brooks, E., Gazeau, M., Sahni, H., Singh, S. and Mnih, V. (2023) 'In-context reinforcement learning with algorithm distillation', in *The Eleventh International Conference on Learning Representations (ICLR 2023)*. OpenReview.net. Also available as arXiv:[2210.14215](https://arxiv.org/abs/2210.14215).

Lattimore, T. and Szepesvári, C. (2020) *Bandit Algorithms*. Cambridge: Cambridge University Press. doi: [10.1017/9781108571401](https://doi.org/10.1017/9781108571401).

Li, Y., Yue, X., Xu, Z., Jiang, F., Niu, L., Lin, B. Y., Ramasubramanian, B. and Poovendran, R. (2025) 'Small models struggle to learn from strong reasoners', in *Findings of the Association for Computational Linguistics: ACL 2025*, pp. 25366–25394. Association for Computational Linguistics, Vienna, Austria. doi: [10.18653/v1/2025.findings-acl.1301](https://doi.org/10.18653/v1/2025.findings-acl.1301).

Lin, B. Y., Ravichander, A., Lu, X., Dziri, N., Sclar, M., Chandu, K., Bhagavatula, C. and Choi, Y. (2024) 'The unlocking spell on base LLMs: Rethinking alignment via in-context learning', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2312.01552](https://arxiv.org/abs/2312.01552).

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C. and Han, S. (2024) 'AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration', in *Proceedings of Machine Learning and Systems 6 (MLSys 2024)*. Preprint version: arXiv:2306.00978, whose title omits “On-Device”.

Liu, R., Geng, J., Peterson, J. C., Sucholutsky, I. and Griffiths, T. L. (2025) 'Large language models assume people are more rational than we really are', in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. OpenReview.net. Also available as arXiv:[2406.17055](https://arxiv.org/abs/2406.17055).

Liu, R., Geng, J., Wu, A. J., Sucholutsky, I., Lombrozo, T. and Griffiths, T. L. (2025) 'Mind your step (by step): Chain-of-thought can reduce performance on tasks where thinking makes humans worse', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 38489–38517. PMLR. Also available as arXiv:[2410.21333](https://arxiv.org/abs/2410.21333).

Liu, R., Sun, Y., Zhang, M., Bai, H., Yu, X., Yu, T., Yuan, C. and Hou, L. (2025) 'Quantization hurts reasoning? An empirical study on quantized reasoning models', in *Second Conference on Language Modeling (COLM 2025)*. OpenReview.net. Also available as arXiv:[2504.04823](https://arxiv.org/abs/2504.04823).

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., Ding, H., Men, K., Yang, K., Zhang, S., Deng, X., Zeng, A., Du, Z., Zhang, C., Shen, S., Zhang, T., Su, Y., Sun, H., Huang, M., Dong, Y. and Tang, J. (2024) 'AgentBench: Evaluating LLMs as agents', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2308.03688](https://arxiv.org/abs/2308.03688).

Liu, B., Yu, S., Liu, Z., Guertler, L., Qi, P., Balcells, D., Liu, M., Tan, C., Shi, W., Lin, M., Lee, W. S. and Jaques, N. (2026) 'SPIRAL: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning', in *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. OpenReview.net. Also available as arXiv:[2506.24119](https://arxiv.org/abs/2506.24119).

Meta (2025) 'Llama models', Official GitHub repository meta-llama/llama-models (model cards, release table and licences for Llama 3.1, 3.2, 3.3 and 4). Available at: <https://github.com/meta-llama/llama-models> (Accessed: 8 October 2026).

Mistral AI (2025a) 'Introducing Mistral 3', Mistral AI news announcement, 2 December 2025 (Mistral Large 3 and the Ministral 3 family). Available at: <https://mistral.ai/news/mistral-3/> (Accessed: 8 October 2026).

Mistral AI (2025b) 'Mistral Small 3', Mistral AI news announcement, 30 January 2025. Available at: <https://mistral.ai/news/mistral-small-3/> (Accessed: 8 October 2026).

Mistral AI (2025c) 'Mistral Small 3.1', Mistral AI news announcement, 17 March 2025. Available at: <https://mistral.ai/news/mistral-small-3-1/> (Accessed: 8 October 2026).

Monea, G., Bosselut, A., Brantley, K. and Artzi, Y. (2025) 'LLMs are in-context bandit reinforcement learners', in *Second Conference on Language Modeling (COLM 2025)*. OpenReview.net. Also available as arXiv:[2410.05362](https://arxiv.org/abs/2410.05362).

Nie, A., Su, Y., Chang, B., Lee, J., Chi, E. H., Le, Q. V. and Chen, M. (2025) 'EVOLvE: Evaluating and optimizing LLMs for in-context exploration', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 46346–46376. PMLR. Also available as arXiv:[2410.06238](https://arxiv.org/abs/2410.06238).

OpenAI (2025) 'gpt-oss-120b & gpt-oss-20b model card', *arXiv preprint* arXiv:2508.10925. Available at: <https://arxiv.org/abs/2508.10925>.

Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H. and Belinkov, Y. (2025) 'LLMs know more than they show: On the intrinsic representation of LLM hallucinations', in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. OpenReview.net. Also available as arXiv:[2410.02707](https://arxiv.org/abs/2410.02707).

Paglieri, D., Cupiał, B., Coward, S., Piterbarg, U., Wołczyk, M., Khan, A., Pignatelli, E., Kuciński, Ł., Pinto, L., Fergus, R., Foerster, J., Parker-Holder, J. and Rocktäschel, T. (2025) 'BALROG: Benchmarking agentic LLM and VLM reasoning on games', in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. OpenReview.net. Also available as arXiv:[2411.13543](https://arxiv.org/abs/2411.13543).

Pomerleau, D. A. (1988) 'ALVINN: An autonomous land vehicle in a neural network', in D. S. Touretzky (ed.) *Advances in Neural Information Processing Systems 1 (NIPS 1988)*, pp. 305–313. Morgan Kaufmann. Proceedings volume published in 1989.

Qwen Team (2025) 'Qwen3', Official GitHub repository QwenLM/Qwen3 (model release notes, including the Qwen3-2507 updates). Available at: <https://github.com/QwenLM/Qwen3> (Accessed: 8 October 2026).

Qwen Team (2026) 'Qwen3.8', Official GitHub repository QwenLM/Qwen3.8 (release notes for Qwen3.5, Qwen3.6 and Qwen3.8). Available at: <https://github.com/QwenLM/Qwen3.8> (Accessed: 8 October 2026).

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S. and Finn, C. (2023) 'Direct preference optimization: Your language model is secretly a reward model', in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Curran Associates, Inc. Also available as arXiv:[2305.18290](https://arxiv.org/abs/2305.18290).

Robinson, J. and Wingate, D. (2023) 'Leveraging large language models for multiple choice question answering', in *The Eleventh International Conference on Learning Representations (ICLR 2023)*. OpenReview.net. Also available as arXiv:[2210.12353](https://arxiv.org/abs/2210.12353).

Ross, S., Gordon, G. and Bagnell, D. (2011) 'A reduction of imitation learning and structured prediction to no-regret online learning', in *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, Proceedings of Machine Learning Research, vol. 15, pp. 627–635. PMLR, Fort Lauderdale, FL, USA. Also available as arXiv:[1011.0686](https://arxiv.org/abs/1011.0686).

Ruoss, A., Pardo, F., Chan, H., Li, B., Mnih, V. and Genewein, T. (2025) 'LMAct: A benchmark for in-context imitation learning with long multimodal demonstrations', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 52303–52344. PMLR. Also available as arXiv:[2412.01441](https://arxiv.org/abs/2412.01441).

Schaeffer, R., Miranda, B. and Koyejo, S. (2023) 'Are emergent abilities of large language models a mirage?', in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Curran Associates, Inc. Also available as arXiv:[2304.15004](https://arxiv.org/abs/2304.15004).

Schmied, T., Bornschein, J., Grau-Moya, J., Wulfmeier, M. and Pascanu, R. (2026) 'LLMs are greedy agents: Effects of RL fine-tuning on decision-making abilities', in *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. OpenReview.net. Also available as arXiv:[2504.16078](https://arxiv.org/abs/2504.16078).

Sclar, M., Choi, Y., Tsvetkov, Y. and Suhr, A. (2024) 'Quantifying language models' sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2310.11324](https://arxiv.org/abs/2310.11324).

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y. and Guo, D. (2024) 'DeepSeekMath: Pushing the limits of mathematical reasoning in open language models', *arXiv preprint* arXiv:2402.03300. Available at: <https://arxiv.org/abs/2402.03300>.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. and Yao, S. (2023) 'Reflexion: Language agents with verbal reinforcement learning', in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Curran Associates, Inc. Also available as arXiv:[2303.11366](https://arxiv.org/abs/2303.11366).

Shojaee, P., Mirzadeh, I., Alizadeh, K., Horton, M., Bengio, S. and Farajtabar, M. (2025) 'The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity', in *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. Curran Associates, Inc. Also available as arXiv:[2506.06941](https://arxiv.org/abs/2506.06941).

Sinha, A., Arun, A., Goel, S., Staab, S. and Geiping, J. (2026) 'The illusion of diminishing returns: Measuring long horizon execution in LLMs', in *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. OpenReview.net. Also available as arXiv:[2509.09677](https://arxiv.org/abs/2509.09677).

Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y. and Shwartz-Ziv, R. (2025) 'Layer by layer: Uncovering hidden representations in language models', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 55854–55875. PMLR. Also available as arXiv:[2502.02013](https://arxiv.org/abs/2502.02013).

Snell, C., Lee, J., Xu, K. and Kumar, A. (2025) 'Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning', in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. OpenReview.net. Preprint version: arXiv:2408.03314, “Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters”.

Sutton, R. S. and Barto, A. G. (2018) *Reinforcement Learning: An Introduction*. Second edn. Adaptive Computation and Machine Learning. Cambridge, MA: MIT Press.

Szot, A., Schwarzer, M., Agrawal, H., Mazoure, B., Talbott, W., Metcalf, K., Mackraz, N., Hjelm, D. and Toshev, A. (2024) 'Large language models as generalizable policies for embodied tasks', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2310.17722](https://arxiv.org/abs/2310.17722).

Tajwar, F., Jiang, Y., Thankaraj, A., Rahman, S. S., Kolter, J. Z., Schneider, J. and Salakhutdinov, R. (2025) 'Training a generally curious agent', in *Proceedings of the 42nd International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 267, pp. 58227–58281. PMLR. Also available as arXiv:[2502.17543](https://arxiv.org/abs/2502.17543).

Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y. and Chen, Y.-N. (2024) 'Let me speak freely? A study on the impact of format restrictions on large language model performance', in *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track*, pp. 1218–1236. Association for Computational Linguistics, Miami, Florida, US. Preprint version: arXiv:2408.02442, “Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models”. doi: [10.18653/v1/2024.emnlp-industry.91](https://doi.org/10.18653/v1/2024.emnlp-industry.91).

Tan, W., Zhang, W., Liu, S., Zheng, L., Wang, X. and An, B. (2024) 'True knowledge comes from practice: Aligning large language models with embodied environments via reinforcement learning', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2401.14151](https://arxiv.org/abs/2401.14151).

Team Olmo, Ettinger, A. et al. (2025) 'Olmo 3', *arXiv preprint* arXiv:2512.13961. Available at: <https://arxiv.org/abs/2512.13961>.

Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H. and Younis, O. G. (2025) 'Gymnasium: A standard interface for reinforcement learning environments', in *Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track*. Curran Associates, Inc. doi: [10.52202/085713-4916](https://doi.org/10.52202/085713-4916).

Walsh, E. P., Soldaini, L., Groeneveld, D., Lo, K. et al. (2025) '2 OLMo 2 Furious (COLM's version)', in *Second Conference on Language Modeling (COLM 2025)*. OpenReview.net. Shorter conference version of the technical report arXiv:2501.00656, “2 OLMo 2 Furious” (Team OLMo et al.).

Wang, X., Hu, C., Ma, B., Röttger, P. and Plank, B. (2024) 'Look at the text: Instruction-tuned language models are more robust multiple choice selectors than you think', in *First Conference on Language Modeling (COLM 2024)*. OpenReview.net. Also available as arXiv:[2404.08382](https://arxiv.org/abs/2404.08382).

Wang, X., Ma, B., Hu, C., Weber-Genzel, L., Röttger, P., Kreuter, F., Hovy, D. and Plank, B. (2024) '“My answer is C”: First-token probabilities do not match text answers in instruction-tuned language models', in *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 7407–7416. Association for Computational Linguistics, Bangkok, Thailand. doi: [10.18653/v1/2024.findings-acl.441](https://doi.org/10.18653/v1/2024.findings-acl.441).

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A. and Zhou, D. (2023) 'Self-consistency improves chain of thought reasoning in language models', in *The Eleventh International Conference on Learning Representations (ICLR 2023)*. OpenReview.net. Also available as arXiv:[2203.11171](https://arxiv.org/abs/2203.11171).

Wang, Z., Wang, K., Wang, Q., Zhang, P., Li, L., Yang, Z., Jin, X., Yu, K., Nguyen, M. N., Liu, L., Gottlieb, E., Lu, Y., Cho, K., Wu, J., Fei-Fei, L., Wang, L., Choi, Y. and Li, M. (2025) 'RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning', *arXiv preprint* arXiv:2504.20073. Available at: <https://arxiv.org/abs/2504.20073>.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J. and Fedus, W. (2022) 'Emergent abilities of large language models', *Transactions on Machine Learning Research*. Available at: <https://openreview.net/forum?id=yzkSU5zdwD>.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V. and Zhou, D. (2022) 'Chain-of-thought prompting elicits reasoning in large language models', in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. Curran Associates, Inc. doi: [10.52202/068431-1800](https://doi.org/10.52202/068431-1800).

Wu, Y., Tang, X., Mitchell, T. and Li, Y. (2024) 'SmartPlay: A benchmark for LLMs as intelligent agents', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2310.01557](https://arxiv.org/abs/2310.01557).

Yang, A., Yang, B., Zhang, B. et al. (2024) 'Qwen2.5 technical report', *arXiv preprint* arXiv:2412.15115. Qwen Team, Alibaba Group. Available at: <https://arxiv.org/abs/2412.15115>.

Yang, A., Li, A., Yang, B., Zhang, B. et al. (2025) 'Qwen3 technical report', *arXiv preprint* arXiv:2505.09388. Qwen Team, Alibaba Group. Available at: <https://arxiv.org/abs/2505.09388>.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2023) 'ReAct: Synergizing reasoning and acting in language models', in *The Eleventh International Conference on Learning Representations (ICLR 2023)*. OpenReview.net. Also available as arXiv:[2210.03629](https://arxiv.org/abs/2210.03629).

Zhao, T. Z., Wallace, E., Feng, S., Klein, D. and Singh, S. (2021) 'Calibrate before use: Improving few-shot performance of language models', in *Proceedings of the 38th International Conference on Machine Learning*, Proceedings of Machine Learning Research, vol. 139, pp. 12697–12706. PMLR. Also available as arXiv:[2102.09690](https://arxiv.org/abs/2102.09690).

Zheng, C., Zhou, H., Meng, F., Zhou, J. and Huang, M. (2024) 'Large language models are not robust multiple choice selectors', in *The Twelfth International Conference on Learning Representations (ICLR 2024)*. OpenReview.net. Also available as arXiv:[2309.03882](https://arxiv.org/abs/2309.03882).

Zheng, X., Li, Y., Chu, H., Feng, Y., Ma, X., Wang, Z., Luo, J., Guo, J., Qin, H., Magno, M. and Liu, X. (2026) 'An empirical study of Qwen3 quantization', *Visual Intelligence*, 4, Article 11. doi: [10.1007/s44267-026-00114-4](https://doi.org/10.1007/s44267-026-00114-4).

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L. and Levy, O. (2023) 'LIMA: Less is more for alignment', in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Curran Associates, Inc. Also available as arXiv:[2305.11206](https://arxiv.org/abs/2305.11206).

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D. and Hou, L. (2023) 'Instruction-following evaluation for large language models', *arXiv preprint* arXiv:2311.07911. Available at: <https://arxiv.org/abs/2311.07911>.
