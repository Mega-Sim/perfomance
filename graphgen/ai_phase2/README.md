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

## Future scope

This package exposes an RL-ready interface so a future policy model can replace the deterministic heuristic in `refine_assignments(...)`.

## Run on `examples/Drawing1.dxf`

```bash
python -m graphgen.ai_phase1.run_phase1_pipeline \
  --dxf examples/Drawing1.dxf \
  --use_phase2 \
  --phase2_mode heuristic
```

Expected outputs include both baseline Phase 1 files and Phase 2 refined files.
