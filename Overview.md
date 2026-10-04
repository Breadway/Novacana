# Novacana ML - Architecture & Training Overview

A deep-dive into how the Novacana AI was built, trained, and iterated from scratch. Covers everything from how the game state becomes a number array, through the neural network design, all the training techniques used across versions, and the planned v6 architecture.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Representing the Game State](#2-representing-the-game-state)
3. [The Abstract Action Space](#3-the-abstract-action-space)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [The Candidate Head](#5-the-candidate-head)
6. [The Training Pipeline](#6-the-training-pipeline)
7. [Imitation Learning](#7-imitation-learning)
8. [Value Head Pretraining](#8-value-head-pretraining)
9. [Candidate Head Pretraining](#9-candidate-head-pretraining)
10. [Reinforcement Learning](#10-reinforcement-learning)
11. [REINFORCE (v1)](#11-reinforce-v1)
12. [Proximal Policy Optimization - PPO (v2+)](#12-proximal-policy-optimization--ppo-v2)
13. [Generalized Advantage Estimation - GAE (v5+)](#13-generalized-advantage-estimation--gae-v5)
14. [Shaped Rewards](#14-shaped-rewards)
15. [Self-Play and the Snapshot Pool](#15-self-play-and-the-snapshot-pool)
16. [AdvancedCpuPlayer Pool (v5+)](#16-advancedcpuplayer-pool-v5)
17. [Monte Carlo Tree Search - MCTS](#17-monte-carlo-tree-search--mcts)
18. [Population-Based Training - PBT](#18-population-based-training--pbt)
19. [Version History and Results](#19-version-history-and-results)
20. [v6 Roadmap](#20-v6-roadmap)

---

## 1. The Problem

Novacana is a 2–4 player card game where players build star systems to win. Teaching a computer to play it well is a reinforcement learning problem: the agent takes actions (playing cards, targeting opponents, discarding), receives a delayed reward (winning or losing), and must learn a strategy from that sparse signal.

The challenge is that the game is:

- **Partially observable** - you can see your own hand but not opponents' hands
- **Multi-player** - not a two-player zero-sum game, so there's no clean Nash equilibrium to converge to
- **Long-horizon** - games can last 100+ turns, so a mistake early on doesn't show up in the reward until much later
- **Combinatorial** - each card can be played onto different targets, making the concrete action space large

---

## 2. Representing the Game State

Before a neural network can do anything, the game state has to be turned into a fixed-size array of floats. This is called **feature engineering** and lives in `ml/src/feature.rs`.

The encoder produces a `[f32; 104]` vector - 104 numbers - from a `GameView` (what one player can observe at decision time).

### Feature layout

| Range | Count | What it encodes |
|-------|-------|-----------------|
| `[0..29]` | 29 | My hand: count of each of the 29 card types, normalised by dividing by 3 |
| `[29..41]` | 12 | My star chart: complete/incomplete system counts, which star types I have, overall fill fraction, eclipse protection status, approaching-win flag |
| `[41..48]` | 7 | Per-star-type fill fraction for my incomplete systems (how close each system is to completion) |
| `[48..52]` | 4 | Supernova component flags: do I hold Fusion / Gravity / Gas, and can I play a Supernova right now? |
| `[52..88]` | 36 | Up to 3 opponents × 12 features each: hand size, complete/incomplete systems, approaching-win flag, eclipse protection, star types present |
| `[88..96]` | 8 | Global: deck size, turn number, Dark Matter Void class |
| `[96..104]` | 8 | Derived: my hand size, am I the active player, opponent count, best opponent complete-system count, game progress, hand-pressure flag, am I leading, deck exhausted |

### Key design choices

**Normalisation.** Raw counts are divided by expected maximums (hand sizes by 3, complete systems by 3, turn number by 400). This keeps all values roughly in `[0, 1]`, which helps gradient-based optimisers converge.

**Derived signals.** Features like "am I leading" (feature 102) and "approaching win" (feature 90) are redundant - they can be computed from other features - but they're included anyway. Neural networks can technically learn these combinations, but having them pre-computed makes the value head's job easier, especially early in training.

**Missing information.** The encoder does NOT capture game history. It's a pure snapshot: the model sees only the current state, not what cards opponents have played over the last 10 turns, or what was discarded. This is a key limitation addressed in v6.

---

## 3. The Abstract Action Space

The game's concrete action space is large: you might be able to play a Discovery card onto any of several systems, each being a different concrete action. Instead of having the network reason over every concrete possibility, we define 35 **abstract actions** - one per strategic category - and handle the "which specific target" question separately (see [The Candidate Head](#5-the-candidate-head)).

```
 0        Pass
 1- 6     PlayStar       (YellowDwarf, RedGiant, RedSupergiant, WhiteDwarf, RedDwarf, BrownDwarf)
 7-11     PlayDiscovery  (AsteroidBelt, DwarfPlanet, RockyPlanet, GasGiant, Comet)
12-17     PlaySolarSystem (same 6 star kinds - starting a NEW system vs extending)
18-22     PlayFlare      (StellarAlignment, CosmicDenial, Wormhole, Ignition, AstralProjection)
23        PlaySupernovas
24-30     PlayFracture   (BlackHole, Athena, Pulsar, DarkMatterVoid, QEF, SNC, Shockwave)
31-32     PlayProtostar  (DrawTwo, SearchWhiteDwarf)
33-34     PlayOmen       (SolarEclipse, PlanetOfLife)
```

Notice PlayStar (extending an existing system) and PlaySolarSystem (starting a new one) are separate abstract actions for the same card, because they're fundamentally different strategic decisions.

### Legal action masking

At any given state, only some actions are legal. The model produces 35 logits, but illegal ones are **masked out** before applying softmax: their probabilities are forced to zero, so the model never samples them. This is done with a 64-bit integer where each bit corresponds to one abstract action.

```
mask = 0b...000101  →  abstract actions 0 and 2 are legal
```

The masked softmax shifts by the maximum legal logit first (for numerical stability), then normalises only over legal actions.

---

## 4. Neural Network Architecture

The network is a **residual MLP** (Multi-Layer Perceptron with skip connections). Defined in `ml/src/net.rs`.

### Body (shared representation)

```
Input:  [f32; 104]   (game state features)
  │
  L1: Linear(104 → 256) + ReLU
  │
  L2: Linear(256 → 256) + ReLU  ← residual: output += L1 output
  │
  L3: Linear(256 → 256) + ReLU  ← residual: output += L2 output
  │
  L4: Linear(256 → 128) + ReLU  ← bottleneck, no residual (dim changes)
  │
 [f32; 128]  ("body" output - shared by all heads)
```

### Heads (task-specific outputs)

From the 128-dim body output, two heads branch off:

```
Policy head:  Linear(128 → 35)         → 35 raw logits (no activation)
Value head:   Linear(128 → 1) + Sigmoid → single float in (0, 1)
```

The **policy head** outputs a probability distribution over the 35 abstract actions after masked softmax is applied. This tells the model *which card type to play*.

The **value head** outputs an estimate of the probability that this player will win from the current state. It's used to compute advantages during RL training (is this outcome better or worse than expected?).

### Why residual connections?

In a plain MLP, gradients flow backward through every layer via the chain rule. Deep networks (4+ layers) can suffer from **vanishing gradients**: by the time the gradient reaches the first layer it's been multiplied together so many times it's nearly zero, and the first layers stop learning.

A residual connection adds a shortcut from the input of a layer to its output:

```
z2 = L2(a1) + a1    (instead of just  z2 = L2(a1))
```

This means gradients can flow directly backward through the skip connection without passing through the weight matrix, keeping the signal alive in deeper layers.

### Weight initialisation

All layers use **He (Kaiming) initialisation**:

```
std = sqrt(2.0 / fan_in)
```

This is specifically designed for ReLU networks. Plain random initialisation (e.g. `N(0, 1)`) tends to either saturate ReLUs (all activations become 0) or explode (activations grow exponentially through layers). He init keeps the variance of activations approximately constant at each layer.

### The Adam optimiser

All training uses **Adam** (Adaptive Moment Estimation), with β1=0.9, β2=0.999, ε=1e-8.

Adam is a gradient descent variant that maintains per-parameter running averages of the gradient (first moment, `m`) and the squared gradient (second moment, `v`). The effective step size for each parameter adapts automatically:

```
m = 0.9 * m + 0.1 * gradient         (exponential moving average of gradient)
v = 0.999 * v + 0.001 * gradient²    (exponential moving average of gradient²)

# Bias-corrected estimates:
m̂ = m / (1 - 0.9^t)
v̂ = v / (1 - 0.999^t)

weight -= lr * m̂ / (sqrt(v̂) + 1e-8)
```

The key insight: parameters that receive rare but large gradients get smaller effective steps (v is large → denominator is large), while parameters with small, consistent gradients get relatively larger steps. This is especially useful in games where some actions are rare.

---

## 5. The Candidate Head

The policy head decides *what type of card to play* (e.g., "play a Discovery card"). But there may be multiple concrete instantiations of that choice - which system to attach the Discovery to, or which star to play. These sub-decisions need to be resolved.

The **candidate head** is a small auxiliary network that scores options against each other. It lives in `ml/src/candidate.rs` and `ml/src/net.rs`.

### Architecture

```
Input:  concat(body_output[128], candidate_features[24]) = [f32; 152]
  │
  lc1: Linear(152 → 32) + ReLU
  │
  lc2: Linear(32 → 1)          → raw score (no activation)
```

For K candidates, the head is run K times (once per option), producing K raw scores. A softmax converts these to probabilities, and the highest-probability option is selected.

### Candidate feature encoding (24 dimensions, `CAND_DIM = 24`)

Each candidate option is encoded into a 24-float vector:

| Range | What it encodes |
|-------|-----------------|
| `[0-4]` | Type flags: is this a card / system / target / etc.? |
| `[5-11]` | Card features: card type one-hot, hand position |
| `[12-16]` | System features: completion fraction, number of discoveries needed |
| `[17-19]` | Player features: my complete systems, best opponent pressure |
| `[20-21]` | Shared: deck progress, turn progress |
| `[22-23]` | Positional: slot index / total options, total / 10 |

### Where the candidate head is used

- **`choose_discards`** - scores each card in hand; discards the highest-scoring ones (model learns "which cards AdvCPU would discard")
- **`pick_targets`** - scores each target option; picks the top N
- **`decide`** - for binary/multi-choice card effects (e.g. Protostar: draw two vs. search), uses positional encoding since option text isn't available as features
- **Concrete action disambiguation** in `choose_action` - when multiple concrete actions map to the same abstract action (e.g. PlayDiscovery(AsteroidBelt) onto system A vs. system B), scores each concrete option and returns the best

### Why separate from the policy?

The policy head is trained with RL signals - it needs to learn long-term strategic value. The candidate head is trained purely by imitation - it learns "what AdvCPU would do" at the sub-decision level. Keeping these separate avoids polluting the RL gradient with sub-decision noise, and allows the candidate head to be pretrained before RL begins.

---

## 6. The Training Pipeline

Training happens in five sequential phases:

```
1. Data Generation    →  play 15,000 games with AdvancedCpuPlayer, record every decision
2. Imitation Train    →  teach the network to copy AdvCPU's actions
3. Candidate Pretrain →  teach the candidate head to copy AdvCPU's sub-decisions
4. Value Pretrain     →  teach the value head to predict game outcomes
5. RL Self-Play       →  improve through self-play with PPO
```

Each phase uses the same `models/net_v5.bin` file, updating it in place. The output of phase 4 is the seed for phase 5; the RL output is written to `models/rl_v5.bin`.

---

## 7. Imitation Learning

**Goal:** give the network a sensible starting policy before RL begins.

RL from random weights is extremely slow in complex games - the agent plays randomly for thousands of games before stumbling on any winning strategies. Imitation learning solves this by training the network to copy a known decent player (the `AdvancedCpuPlayer` heuristic bot).

### How it works

Generate 15,000 games where every player is `AdvancedCpuPlayer`. At each decision, record:
- The 104-feature game state
- The legal action mask
- Which abstract action AdvCPU chose
- The final outcome (win=1.0, loss=0.0)

This produces roughly 4.26 million training samples. Then train the network on this data using **cross-entropy loss**:

```
policy_loss = -log(probability assigned to AdvCPU's chosen action)
```

The idea: make the network's probability distribution over actions match what AdvCPU would do. If AdvCPU played a RedDwarf star, we want the network to assign high probability to abstract action 5 (PlayStar/RedDwarf).

The value head is also trained jointly here, with **MSE loss** against the win/loss outcome:

```
value_loss = (predicted_win_probability - actual_outcome)²
```

### Why imitation first?

- Much faster convergence than RL from scratch
- Avoids catastrophic early RL exploration (random plays, wasting all samples on terrible strategies)
- Gives the value head meaningful signal to bootstrap from
- AdvCPU is a reasonable player, so the imitation baseline is already competitive

The downside: the network can only be as good as AdvCPU through imitation alone. RL is needed to exceed the teacher.

---

## 8. Value Head Pretraining

After imitation, the value head is trained again in a dedicated **value pretraining** phase. This time, only the body and value head weights are updated - the policy head is left alone.

### Why a separate phase?

During imitation training, the policy and value losses are optimised jointly. The network prioritises the policy loss (it has 35 outputs, generating lots of gradient) and the value head can get undertrained.

A dedicated value pretraining pass specifically optimises the body representation for predicting win probability. A well-calibrated value head is critical for RL: if the value head is wrong, the advantage estimates used in PPO will be noisy, making learning unstable.

---

## 9. Candidate Head Pretraining

The candidate head is trained in its own phase using the candidate samples collected during data generation (4.72 million samples in v5).

### How it works

Each sample has:
- A 104-feature game state
- K encoded candidates (cards, systems, or options to choose from)
- Which candidate AdvCPU actually chose

The training objective is **softmax cross-entropy over the candidates**:

```
cand_loss = -log(score_probability[chosen_idx])

where score_probabilities = softmax(candidate_head_scores for all K options)
```

Crucially, only the candidate head layers (`lc1`, `lc2`) are updated. The body weights are frozen as context - the candidate head learns to read the game situation from the body output, but doesn't corrupt the body representation.

### Why this matters (the v4 lesson)

In v4, the `pretrain-candidates` step didn't exist yet when RL training started. The candidate head had random weights, so every `choose_discards`, `pick_targets`, and `decide` call during self-play made effectively random sub-decisions. This was far worse than AdvCPU's heuristics for those choices, and the noise from random sub-decisions corrupted the RL training signal so badly that v4 regressed all the way back to near-random (26.6%) - worse than v1 (24.5%).

The fix in v5: always pretrain the candidate head before RL begins.

---

## 10. Reinforcement Learning

After pretraining, the network plays games against itself (and against AdvancedCpuPlayer as a baseline opponent), using the outcomes to improve beyond the imitation-learning baseline.

The core loop per RL iteration:

```
1. Collect 2,000 self-play games using the current policy (stochastically)
2. Compute returns for each decision using GAE
3. Run 4 passes of PPO gradient updates over the collected data
4. Save checkpoint
```

Over 300 iterations, this accumulates 600,000 games of experience.

---

## 11. REINFORCE (v1)

The simplest RL algorithm. For each decision in a game:

```
loss = -log π(action | state) × return
```

Where `return` is the discounted sum of future rewards from that decision onward (just the final win/loss outcome discounted by γ^t).

The intuition: if we won, increase the probability of every action we took (they were "probably good"). If we lost, decrease them. Scale by the return magnitude.

**Problem:** REINFORCE has extremely high **variance**. In a 4-player game with 100+ decisions, most individual decisions have nearly zero impact on the final outcome. Assigning credit (or blame) equally to all decisions based on the final result is very noisy. A good play might be penalised because the team eventually lost; a bad play might be rewarded because it happened in a winning game.

v1 used REINFORCE and achieved **24.5%** win rate - just below the 25% random baseline.

---

## 12. Proximal Policy Optimization - PPO (v2+)

PPO is a more sophisticated RL algorithm that addresses two major problems with REINFORCE:

1. **High variance** - solved with a value baseline (advantage estimation)
2. **Unstable updates** - solved with clipped policy ratios

### Advantage estimation

Instead of using the raw return as the learning signal, PPO uses the **advantage**:

```
advantage = return - V(state)
```

Where `V(state)` is the value head's estimate of how good this state is. The advantage asks: "was this outcome *better or worse than expected*?" This dramatically reduces variance because:

- If you're in a winning position (V=0.8) and you win (return=1.0), the advantage is only 0.2 - the win was expected
- If you're in a losing position (V=0.2) and you win (return=1.0), the advantage is 0.8 - this was a surprisingly good outcome

### Clipped policy ratio

The core PPO update prevents the policy from changing too drastically in one step. It computes the ratio of the new policy's probability to the old (data-collection) policy's probability:

```
ratio = π_new(action | state) / π_old(action | state)
      = exp(log π_new - log π_old)
```

Then clips this ratio to `[1 - ε, 1 + ε]` (with ε=0.2) and uses the clipped or unclipped loss, whichever is smaller:

```
L_PPO = min(ratio × advantage,  clip(ratio, 1-ε, 1+ε) × advantage)
```

If the ratio is already outside `[0.8, 1.2]` and trying to move further in the beneficial direction, the gradient is zeroed - no update. This acts as a **trust region**: don't take too large a policy step in any single update, because the data was collected under the old policy and might not be representative anymore.

### Multiple epochs per batch

Unlike REINFORCE which makes one gradient pass per batch of experience, PPO reuses each batch for **4 gradient passes**. This improves sample efficiency - you extract more learning signal from the same games - while the clipping prevents over-fitting to any particular batch.

### Entropy bonus

An entropy term is added to the loss:

```
entropy = -Σ p(a) × log p(a)    (Shannon entropy over action probabilities)
```

Maximising entropy encourages the policy to stay exploratory - not collapse to always playing the same few actions. Without this, the policy can prematurely converge to a suboptimal strategy.

---

## 13. Generalized Advantage Estimation - GAE (v5+)

GAE is a technique for computing lower-variance advantage estimates. It addresses the fundamental trade-off between bias and variance in return estimation.

### The problem

- **Monte Carlo returns** (full game played out): low bias (uses actual outcomes), high variance (noisy due to long game horizon)
- **1-step TD returns** (`r + γV(s')−V(s)`): low variance, high bias (relies entirely on the value function being accurate)

GAE interpolates between these extremes using a hyperparameter λ ∈ [0, 1]:

```
δ_t     = r_t + γ × V(s_{t+1}) − V(s_t)    (1-step TD error)
A_t     = δ_t + (γλ) × A_{t+1}              (recursive GAE)
Return_t = A_t + V(s_t)                      (used as the training target)
```

With λ=0.95 (used in v5):
- High λ → closer to Monte Carlo (more variance, less bias)
- Low λ → closer to 1-step TD (less variance, more bias)

In practice, λ=0.95 gives excellent results across many RL tasks. The key improvement over plain returns: the value function acts as a **variance reduction baseline** at every step, not just the final outcome.

### Why this matters

In a Novacana game with 100 decisions, a plain discounted return of γ^99 ≈ 0.37 at the first decision heavily dilutes the signal. GAE re-uses the value function's prediction at every step, so each decision's advantage is estimated relative to the immediate neighbourhood, not the game-end outcome 99 steps later.

---

## 14. Shaped Rewards

The game's natural reward is 1.0 for winning and 0.0 for losing - delivered only at game end. With 100+ decisions per game and γ=0.99, rewards from early decisions are heavily discounted. To give the network denser feedback, **shaped rewards** are added at intermediate steps:

| Event | Reward |
|-------|--------|
| Completing a star system | +0.10 |
| Crossing the 3-system win threshold | +0.15 bonus |
| Opponent completes a system | −0.05 |

These are heuristic and require care - aggressive shaping can cause reward hacking (the agent optimises for intermediate rewards instead of actually winning). The magnitudes here are kept small relative to the final 1.0/0.0 outcome so the net strategic goal isn't distorted.

---

## 15. Self-Play and the Snapshot Pool

During RL, the agent plays against past versions of itself rather than a fixed opponent. This prevents overfitting to one opponent's strategy.

### Snapshot pool

Every 20 iterations, the current network weights are saved to a pool (max 10 snapshots). When generating a game:
- 50% chance: one learner seat vs 3 snapshot opponents (or AdvCPU)
- 50% chance: all 4 seats are learners

Pool opponents are randomly sampled from the historical snapshots. This ensures the agent trains against a diverse mix of past selves, not just the most recent one.

### Why this helps

Without a pool, the agent often falls into **strategy cycling**: it beats opponent strategy A, then overspecialises against A, becomes vulnerable to B, trains against B, becomes vulnerable to A again. The pool maintains "memory" of past strategies, stabilising training.

---

## 16. AdvancedCpuPlayer Pool (v5+)

In v5, `AdvancedCpuPlayer` (the deterministic heuristic bot) is permanently present in the opponent pool with a 30% sampling probability.

The purpose is to maintain **baseline pressure**. Early in RL training, snapshot opponents are weak (untrained or barely trained networks). Without AdvCPU in the pool, the agent might learn to exploit early-training weaknesses that don't exist in a real game. AdvCPU provides a consistent floor of strategic competence, ensuring the agent never learns strategies that only work against confused opponents.

---

## 17. Monte Carlo Tree Search - MCTS

MCTS is a search algorithm used at evaluation time (and in v6, at training time). It's available via the `--mcts N` flag when running `eval`.

### How it works

At each decision, instead of immediately picking the highest-probability action, MCTS runs N **simulations** from the current state:

1. **Selection**: use UCT (Upper Confidence bound for Trees) to pick which action to simulate next, balancing exploitation (high-value actions) vs exploration (under-sampled actions):
   ```
   UCT(a) = Q(a) + C × prior(a) × sqrt(total_visits) / (1 + visits(a))
   ```

2. **Simulation**: clone the game state, force the chosen action, then have all players act greedily with the policy network until the game ends

3. **Backpropagation**: record whether the simulation resulted in a win for our player; update the action's visit count and value estimate

4. **Final selection**: pick the action with the **most visits** (not highest estimated value - visits are more robust to outliers)

### Perfect Information Monte Carlo (PIMC)

Because opponent hands are hidden, MCTS can't clone the *true* game state - it only has `GameView`. The approach used is **determinization**: sample plausible hidden cards for opponents (uniform over unseen cards), run the simulation in that hypothetical world, and average over many such worlds. This is approximate but works well in practice.

### MCTS vs pure policy

MCTS is slower (each decision requires N full game simulations) but stronger. It effectively uses the policy network as a fast evaluator/rollout policy, while the tree search provides lookahead. The trade-off is roughly:

- Policy alone: ~150 games/sec, uses learned strategy only
- MCTS-50: much slower, but can find strong plays the policy would miss

---

## 18. Population-Based Training - PBT

PBT trains a **population** of N agents in parallel, periodically replacing the worst-performing agents with perturbed copies of the best-performing ones.

```
For each iteration:
  1. Each agent generates self-play games (against the rest of the population)
  2. Each agent trains for one gradient step

Every 20 iterations:
  3. Evaluate each agent against AdvancedCpuPlayer (100 games)
  4. Rank by win rate
  5. Replace the bottom 50% with perturbed copies of the top 50%

Perturbation: multiply each weight by (1 + ε), where ε ~ Uniform[-0.02, 0.02]
```

PBT serves as automatic hyperparameter tuning - well-performing agents' hyperparameters (learning rate, temperature, etc.) are inherited by the population, and small perturbations explore nearby hyperparameter space. It also naturally maintains diversity: different agents specialise in different strategies, and their games against each other provide richer training signal than any single agent playing alone.

---

## 19. Version History and Results

### v1 - REINFORCE baseline (24.5%)

**Architecture:** 104 → 256 → 256 → 256 → 128 → {35, 1}

**Training:** Imitation learning + REINFORCE RL

**What we learned:** REINFORCE's high variance completely overwhelmed any learning signal. With no value baseline, 4-player credit assignment is effectively impossible. The result (24.5%) was slightly below the 25% random baseline, meaning REINFORCE actively hurt the imitation-learned policy through noise.

---

### v2 - Value pretraining (45%)

**What changed:** Added a dedicated value pretraining phase before RL.

**Why it worked:** A pretrained value head dramatically reduces RL variance. With an accurate `V(s)`, the advantage `return − V(s)` is small for expected outcomes and large only for surprising ones - exactly the signal we want. This single addition jumped performance from 24.5% to ~45%, close to beating AdvCPU (which would be >50%).

---

### v3 - PPO + shaped rewards + snapshot pool (43.8%)

**What changed:** Replaced REINFORCE with PPO. Added shaped rewards. Added snapshot pool.

**Why the result was similar to v2:** PPO is theoretically superior but requires more careful tuning. The shaped rewards and snapshot pool helped, but may have been offset by PPO hyperparameters not yet dialled in. 43.8% vs 45% is within the noise margin of a 500-game eval (±2-3%).

---

### v4 - Candidate head (26.6% - regression)

**What changed:** Added the candidate head architecture and wired it into all sub-decisions (`choose_discards`, `pick_targets`, `decide`).

**What went wrong:** The `pretrain-candidates` step didn't exist yet when v4's RL training started. The candidate head had **random weights** and was actively used for every discard, target, and decide call during self-play. Random sub-decisions are significantly worse than AdvCPU's heuristics for those choices, generating extremely noisy training data. The RL signal was so corrupted that the model regressed from 43.8% all the way to 26.6% - worse than v1.

**The lesson:** always pretrain sub-decision heads before RL. Never run a randomly-initialised head in the hot path of self-play.

---

### v5 - Correct pretraining order (in progress)

**What changed:**
- **Candidate pretraining step** added before RL (the v4 fix)
- **GAE** (λ=0.95) replacing plain discounted returns
- **AdvancedCpuPlayer** permanently in the opponent pool (30% share)
- 300 RL iterations (up from ~200 in earlier versions)
- 15,000 games for data generation (3× more than previous)

**Expected improvement:** At minimum, recovering v3's 43.8% is expected. The combination of properly-pretrained candidate head + GAE lower variance + AdvCPU baseline pressure should push into the 45–55% range.

**Training pipeline:**

```bash
./novacana-ml generate        --games 15000 --out data/train_v5.bin --cand-out data/candidates_v5.bin
./novacana-ml train           --data data/train_v5.bin --out models/net_v5.bin --epochs 50
./novacana-ml pretrain-candidates --data data/candidates_v5.bin --model models/net_v5.bin --epochs 30
./novacana-ml pretrain-value  --data data/train_v5.bin --model models/net_v5.bin --epochs 20
./novacana-ml rl              --model models/net_v5.bin --out models/rl_v5.bin \
  --iters 300 --games 2000 --lambda 0.95 --advcpu-share 0.3 \
  --ppo-epochs 4 --clip-eps 0.2 --entropy 0.01
./novacana-ml eval            --model models/rl_v5.bin --games 500
```

---

## 20. v6 Roadmap

v6 is planned and locked in. Three major architectural changes and one training paradigm shift.

### Change 1: Extended feature vector (~104 → ~160 dimensions)

Add historical information to the per-decision snapshot in `feature.rs`:

- Recent card plays per player (last 5 plays, encoded as card-type flags, ~20 features)
- Binary flags: has each opponent played a Flare / Fracture / Supernova yet in this game
- Discard pile composition summary

This is purely additive - no architecture changes, just a wider input. It gives the model access to information human players actively track ("opponent has already played their Wormhole, so they can't respond to my chain").

### Change 2: Stacked recent states (3 × feature_dim input)

Instead of a single 160-feature snapshot, concatenate the last 3 encoded game states as the network input (~480 floats). This gives the MLP temporal awareness without needing recurrence.

The rolling state buffer is maintained in `RecordingPlayer` and `SelfPlayPlayer` - each player tracks their last 3 states, and the oldest is dropped when a new one arrives.

A full recurrent architecture (LSTM/GRU) or transformer over the event log would be more expressive but significantly harder to train and would require GPU acceleration to run at speed. Stacked states is a practical middle ground: simple to implement, provides temporal context, stays within the existing MLP framework.

### Change 3: 512-wide network

Increase all hidden layers from 256 to 512 wide:

```
~480 → 512 → 512(+res) → 512(+res) → 256 → {35, 1}
```

At this scale, GPU training becomes worthwhile. The current 256-wide network on CPU is fast enough; at 512 wide the matrix multiplications are 4× larger, and a GPU (like Hestia's Intel Arc A380) would begin to provide meaningful speedup, especially combined with the larger input dimension.

### Change 4: AlphaZero-style training loop

This is the highest-impact change. Currently, PPO learns from win/loss outcomes - a sparse signal delivered at the end of a 100+ turn game. AlphaZero learns from **MCTS visit distributions** - a dense signal at every single decision.

The AlphaZero training loop works as follows:

1. **Data generation:** at every decision during self-play, run N MCTS simulations and record the normalised visit count distribution across actions as the policy target
   ```
   target_policy[a] = visits(a) / total_visits
   ```

2. **Policy training:** instead of cross-entropy against a single sampled action, train the policy head to match the entire MCTS distribution (soft targets):
   ```
   loss = -Σ target_policy[a] × log network_policy[a]    (KL divergence)
   ```

3. **Value training:** unchanged - predict win probability from the game outcome

Why this is so powerful: MCTS is a **policy improvement operator**. Given a policy π, running MCTS produces a better policy π' (the visit distribution). Training the network to match π' makes the next generation of MCTS even better. This recursive self-improvement is why AlphaZero-style approaches achieved superhuman play in Go and Chess.

The key difference from standard PPO: every decision in every game provides a full distribution as a training signal, not just one sampled action and a distant outcome. Credit assignment nearly solves itself because the MCTS value estimates provide accurate per-decision feedback.

### Expected performance

| Improvement | Estimated gain |
|-------------|---------------|
| Extended features (~160 dim) | +3–5% |
| Stacked 3 states | +5–8% |
| AlphaZero training loop | +10–20% |
| 512-wide network | +2–5% |
| **Combined target** | **55–65%** |

Beating AdvancedCpuPlayer consistently (>50% win rate against 3 of them) is the first milestone. Beyond that, the ceiling is set by the fundamental difficulty of multi-player imperfect-information games - there's no Nash equilibrium to converge to the way there is in two-player zero-sum games like Chess or Go.

### What v7 might look like

A full transformer over the game event log (each card play, system completion, chain trigger as a token) would be the next step after AlphaZero. This requires GPU training to be practical at scale, but would allow the model to attend to any point in the game history, learn opponent "styles", and reason about long-range cause and effect. That's the architecture that would push Novacana AI toward genuinely sophisticated strategic play.
