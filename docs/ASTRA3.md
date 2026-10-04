# v3

Same network as v2. The diagram, the 104-float input, and the parameter count are in `docs/ASTRA2.md`. Weights are `models/net_v3.bin`, `models/rl_v3.bin`, and `ml/models/net_v3.bin`.

v2 had already added a value-pretraining pass and reached about 45% with REINFORCE. v3 kept that net and changed the training around it.

- PPO replaced REINFORCE. The signal is the advantage, the return minus the value head's estimate. The policy ratio is clipped at 0.2, and each batch of games is reused for 4 passes. An entropy bonus is added so the policy does not collapse onto a few actions.
- Shaped rewards sit on the terminal win or loss: +0.10 for completing a system, +0.15 for crossing three complete systems, and -0.05 when an opponent completes one.
- A snapshot pool supplies opponents. Weights are saved every 20 iterations and 10 snapshots are kept. Half the games are one learner against three pool opponents. The other half are four learners.

The recorded eval is 43.8% over 500 games. That sits inside the 2-3% noise of v2's 45% on the same eval size.
