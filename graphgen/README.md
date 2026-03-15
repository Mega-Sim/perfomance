# graphgen

표준 레이아웃 spec 로드와 validator 실행 예시:

```bash
python -c "from graphgen.spec import load_standard_spec; print(load_standard_spec()['spec_version'])"
python -m graphgen.tools.validate_standard_dataset
```

AI 연구용(운영 DXF 경로와 분리) 최소 실험 scaffold:

```bash
python -m graphgen.ai_training.prepare_dataset --out_dir datasets/standard/training
python -m graphgen.ai_training.train_segmentation_baseline --manifest datasets/standard/training/manifest.jsonl --eval_split val
```

상세 배경/설계는 `docs/ai_training_plan.md`를 참고하세요.
