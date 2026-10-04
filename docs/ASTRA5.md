# v5

Same network as v4. The diagram, the candidate head, and the parameter count are in `docs/ASTRA4.md`. Weights are `models/net_v5.bin` and `models/rl_v5.bin`.

v4 had wired that candidate head into discards, target picks, and decide prompts, then started RL while the head was still random. Self-play sub-decisions were noise, and the run finished at 26.6%. v5 keeps the net and fixes the schedule around it.

- The candidate head is pretrained on AdvCPU's sub-decisions before RL. Only the candidate layers train. The body stays frozen.
- Advantages use GAE with lambda 0.95, in place of one discounted return from the end of the game.
- AdvancedCpuPlayer stays in the opponent pool and is drawn 30% of the time, so early self-play is not only against weak snapshots.
- Imitation data is 15,000 games, three times the earlier generation. RL runs 300 iterations of 2,000 games, up from about 200 iterations.
- Order: imitation for 50 epochs, candidate pretrain for 30, value pretrain for 20, then PPO with clip 0.2, 4 passes per batch, and entropy coefficient 0.01.

`Overview.md` leaves this run in progress. It expected at least a return to v3's 43.8%, and hoped for 45-55%. It does not record a finished win rate.
