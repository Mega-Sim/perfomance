# AI Phase 2: RL-ready graph refinement scaffold

## Why Phase 2 is placed after `solve()`

The production path is already stable as:

`DXF -> parse/build graph -> solve directions -> render/dump outputs`

Phase 2 is intentionally inserted **after `solve()`** so it can refine an already-directed graph without changing Phase 1 graph construction or solver internals.

## Current scope (minimal)

- Direction refinement only (`keep` / `flip` per existing edge)
- No topology edits:
  - no node split/merge
  - no edge add/delete

## RL baseline scope in this repo

- `rl_env.py`: Gymnasium env (`keep`/`flip` binary action per candidate edge)
- `train_phase2_rl.py`: Stable-Baselines3 PPO trainer
- `refine_graph.py`: inference mode `ppo` + default fallback mode `heuristic`

This is the first learned-policy baseline (MLP-friendly fixed vector observation), not the final architecture.

## Run on `examples/Drawing1.dxf`

```bash
python -m graphgen.ai_phase1.run_phase1_pipeline \
  --dxf examples/Drawing1.dxf \
  --use_phase2 \
  --phase2_mode heuristic
```

Expected outputs include both baseline Phase 1 files and Phase 2 refined files.

Train PPO (single-layout smoke baseline):

```bash
python -m graphgen.ai_phase2.train_phase2_rl \
  --dxf examples/Drawing1.dxf \
  --total_timesteps 5000 \
  --model_out outputs/models/phase2_ppo_Drawing1 \
  --seed 42
```

Train PPO (multi-layout baseline):

```bash
python -m graphgen.ai_phase2.train_phase2_rl \
  --dxf_dir examples \
  --train_ratio 0.8 \
  --total_timesteps 10000 \
  --model_out outputs/models/phase2_ppo_multilayout \
  --seed 42
```

Alternative input: `--dxf_list <manifest.txt|manifest.json>`.

Use PPO inference:

```bash
python -m graphgen.ai_phase1.run_phase1_pipeline \
  --dxf examples/Drawing1.dxf \
  --use_phase2 \
  --phase2_mode ppo \
  --phase2_model outputs/models/phase2_ppo_Drawing1.zip \
  --phase2_seed 42
```

Evaluate baseline vs heuristic vs PPO (and save side-by-side artifacts):

```bash
python -m graphgen.ai_phase2.eval_phase2_rl \
  --dxf examples/Drawing1.dxf \
  --ppo_model outputs/models/phase2_ppo_Drawing1.zip \
  --out_dir outputs/phase2_eval \
  --seed 42
```

Batch evaluate multiple layouts and save aggregate report:

```bash
python -m graphgen.ai_phase2.eval_phase2_batch \
  --dxf_dir examples \
  --ppo_model outputs/models/phase2_ppo_multilayout.zip \
  --out_dir outputs/phase2_batch_eval \
  --seed 42
```

This PPO path remains a practical baseline; multi-layout results may still be worse than heuristic.
