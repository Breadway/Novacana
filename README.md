# Novacana

Self-play ML training engine for a card/board game AI (working name ASTRA), built as
a Rust workspace: game simulation, self-play data generation, and model training/eval.

Cargo workspace members:

- **engine** - game simulation / self-play
- **client**, **server** - play/serve the game
- **ml**, **ml7** - model training and evaluation

See `ASTRA7.md` for the current implementation plan and `GPU_RENTAL.md` for the
training infrastructure setup.

## Building

```bash
cargo build --release
```
