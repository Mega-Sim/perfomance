# Mega-Sim / perfomance

DXF 기반 OHT 레이아웃에서 **그래프를 만들고 방향을 결정**하는 파이프라인입니다.

## Current project status

- ✅ **DXF → Graph Builder** (direct parsing)
- ✅ **Deterministic rule-based direction solver** (기본/안정 경로)
- 🧪 **Phase 2 RL scaffold (experimental)**: rule 결과를 local repair로 보정하는 연구용 구조
- 🗄️ Legacy image segmentation + skeleton pipeline: `graphgen/ai_phase1_legacy/`

> RL 경로는 실험용이며, 기본 동작은 항상 rule 기반입니다.

## Pipeline

`DXF -> Graph -> Direction (rule | rl-repair) -> DXF/Simulator artifacts`

- `rule` (default): 기존 deterministic solver 경로
- `rl-repair` (opt-in): rule solve 이후 국소 방향 뒤집기/수정 시도

## Install

```bash
pip install -r requirements.txt
```

## Usage

### 1) Rule-based (default/stable)

```bash
python -m graphgen.run_phase2 --dxf examples/Drawing1.dxf --mode rule
```

### 2) RL-repair (experimental, opt-in)

```bash
python -m graphgen.run_phase2 --dxf examples/Drawing1.dxf --mode rl-repair
```

### 3) Train minimal RL scaffold

```bash
python -m graphgen.ai_phase2.train --episodes 120
```

### 4) Evaluate rule/random/greedy/rl-repair

```bash
python -m graphgen.ai_phase2.eval
```

Evaluation JSON output:
- `results/phase2_eval_*.json`

## Experimental vision scaffold (minimal, first PR)

This repository now includes a **small trainable image-learning scaffold** in:
- `graphgen/vision/dataset.py`
- `graphgen/vision/model.py`
- `graphgen/vision/train.py`
- `graphgen/vision/infer.py`

Current setup uses `datasets/standard/images` and generates **weak/pseudo masks on-the-fly** from color-threshold rules for initial supervised training.

Train:
```bash
python -m graphgen.vision.train --images_dir datasets/standard/images --epochs 5 --out_dir outputs/vision
```

Infer to graph JSON (single image):
```bash
python -m graphgen.vision.infer \
  --checkpoint outputs/vision/tiny_unet.pt \
  --image datasets/standard/images/0001.png \
  --out outputs/vision/graph_0001.json
```

Infer with random-initialized weights (no checkpoint):
```bash
python -m graphgen.vision.infer \
  --image datasets/standard/images/0001.png \
  --out outputs/vision/graph_0001_random.json
```

> Labels in this vision scaffold are currently weak/pseudo (rule-derived), and this is intentional for early pipeline validation.
> Output graph is a simple first-pass extraction from predicted mask components (for review/iteration).

### Drawing1 image-based AI inference demo

DXF를 렌더링해서 이미지 기반 AI 추론으로 간단 그래프 JSON을 생성합니다.

Render `Drawing1.dxf` to image:
```bash
python -m graphgen.vision.render_dxf --dxf examples/Drawing1.dxf --out tmp/drawing1.png
```

Infer graph from rendered image:
```bash
python -m graphgen.vision.infer --image tmp/drawing1.png --out outputs/drawing1_graph.json
```

One-command pipeline:
```bash
python -m graphgen.vision.run_drawing1_pipeline --dxf examples/Drawing1.dxf --tmp_image tmp/drawing1.png --out outputs/drawing1_graph.json
```

## Notes on Phase 2

- Rule solver를 대체하지 않고, 후처리(local repair)로만 사용
- 액션: edge direction flip (필요 시 assign)
- 보상: nonholonomic/split-merge/tangent 위반 감소 중심
- synthetic graph 세트로 빠른 학습/회귀 테스트 가능

## Legacy Phase 1 image path

이미지 기반 컬러 세그멘테이션/스켈레톤 방식은 아카이브 상태이며 여기 있습니다:
- `graphgen/ai_phase1_legacy/`

## Roadmap (short)

1. RL direction policy 개선 (generalization + robust reward shaping)
2. Traffic / Scheduler integration
3. Multi-OHT simulation and full-system evaluation
