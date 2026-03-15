# Standard Layout AI Training Plan (DXF 운영 경로 분리)

## 1) 현재 상태 진단

### 확인한 실제 저장소 상태
- 운영 파이프라인은 `graphgen/ai_phase1/run_phase1_pipeline.py` 기준 **DXF direct path**이며, 이미지 역추출을 사용하지 않습니다.
- `datasets/standard/`에는 현재 `images/*.png` 5장과 `spec.json`, `labels.template.jsonl`만 있으며, `graphs/`, `manifest.jsonl`, `labels.jsonl`은 아직 비어 있거나 미생성 상태입니다.
- `graphgen/ai_phase1_legacy/`는 heuristic + skeleton 기반의 연구/회귀용 코드이며 README에서도 deprecated로 명시되어 있습니다.

### 불일치/갭
- 기존 spec(v1.1)은 `node_marker`를 단일 red로 정의하지만, 실제 validator/legacy는 split(red) + merge(purple)를 모두 사용합니다.
- direction marker는 README에서 검정 화살표로 정의되나, 이전 spec에는 `same_as_track`으로 기록되어 있었고 validator는 black threshold를 사용했습니다.
- legacy color_model은 `spec["phase1"]["color_thresholds"]`를 읽도록 되어 있으나 이전 spec에는 해당 섹션이 없어 실질적으로 threshold가 비어 있었습니다.
- graph supervision 경로(`datasets/standard/graphs`)가 spec에만 있고 실제 샘플은 없어, 현재는 **full graph supervision 학습**을 즉시 시작하기 어렵습니다.

---

## 2) 학습 목적 정의

### 선택: Hybrid pipeline (segmentation + topology extraction)
현재 저장소 구조에서 가장 현실적인 문제 정의는 다음 2단 분리입니다.

1. **학습 목적(vision model)**: semantic segmentation / marker detection.
   - 현재 이미지와 색상 표준만으로 supervision을 바로 만들 수 있습니다.
2. **추론 목적(graph reconstruction)**: segmentation 결과와 방향 marker를 이용해 topology를 복원.
   - full graph GT가 준비되면 end-to-end graph reconstruction 평가로 확장합니다.

이 구성이 맞는 이유:
- 현재 즉시 존재하는 정답 근거는 이미지의 표준 색상 규칙(spec)입니다.
- graph json GT는 아직 부족하므로, 먼저 segmentation/marker 일반화를 안정화한 뒤 topology 단계로 가는 것이 실현 가능성이 높습니다.
- 운영 DXF path와 충돌 없이 연구 경로를 독립적으로 유지할 수 있습니다.

---

## 3) 수정/추가 파일 목록

- 수정: `datasets/standard/spec.json`
  - class 정의를 split/merge/station/direction으로 명확화.
  - training/split/augmentation/evaluation 섹션 추가.
  - legacy 호환용 `phase1.color_thresholds` 추가.
- 수정: `graphgen/tools/validate_standard_dataset.py`
  - class mask 계산을 spec 기반(split/merge/direction 포함)으로 정합화.
- 추가: `graphgen/ai_training/data_utils.py`
  - class mask 생성, index mask 변환, deterministic split 유틸.
- 추가: `graphgen/ai_training/prepare_dataset.py`
  - `images -> masks + manifest` 학습 입력 생성 entrypoint.
- 추가: `graphgen/ai_training/train_segmentation_baseline.py`
  - 최소 RGB prototype segmentation baseline (연구 scaffold).
- 추가: `graphgen/ai_training/eval_topology_metrics.py`
  - graph JSON 기준 topology metric 계산 스크립트.
- 추가: `graphgen/ai_training/__init__.py`
- 추가: `docs/ai_training_plan.md` (이 문서)

---

## 4) 새 label / split / metric 체계

### Label 체계
- 즉시 생성 가능(supervision 가능)
  - `segmentation_mask` (6-class): background, track, split_marker, merge_marker, station_marker, direction_marker
  - `node_points`는 marker class의 connected component 중심점으로 파생 가능
- graph json 존재 시 추가 가능
  - edge polyline
  - edge direction
  - adjacency/topology consistency

### Split 체계
- `stem` + `seed` 기반 해시 분할(`hash_by_stem`): 재현 가능.
- 기본 비율: train 0.7 / val 0.15 / test 0.15.

### Metric 체계
- segmentation: macro IoU, macro F1.
- topology: node precision/recall, edge recovery rate, direction accuracy, topology consistency.
- 현재 저장소 현실에 맞춰, topology metric은 graph GT가 있는 샘플에 한해 계산.

---

## 5) 실행 방법

```bash
# 1) 표준 데이터 검증
python -m graphgen.tools.validate_standard_dataset

# 2) 학습용 manifest/mask 생성
python -m graphgen.ai_training.prepare_dataset --out_dir datasets/standard/training

# 3) 최소 segmentation baseline 학습/평가(val)
python -m graphgen.ai_training.train_segmentation_baseline \
  --manifest datasets/standard/training/manifest.jsonl \
  --eval_split val \
  --out outputs/ai_training/segmentation_baseline_metrics.json

# 4) (선택) graph gt/pred 비교 metric
python -m graphgen.ai_training.eval_topology_metrics --gt <gt_graph.json> --pred <pred_graph.json>
```

---

## 6) 아직 구현하지 않은 것과 다음 단계

- 미구현
  - neural model 학습(UNet 등) 자체는 아직 미포함.
  - graph reconstruction model/end-to-end training은 GT graph 축적 전까지 보류.
  - augmentation 실제 적용 파이프라인(현재는 spec 정책 정의 + scaffold 단계).
- 다음 단계
  1. `datasets/standard/graphs/*.json` 최소 n>=30 샘플 구축.
  2. marker point/edge polyline intermediate label exporter 추가.
  3. segmentation 모델 학습 코드(torch 등)와 topology decoder 실험을 별도 연구 폴더에서 확장.

---

## 7) 운영 DXF path vs 연구 AI path 분리

- 운영(Production): 기존 그대로 `graphgen/ai_phase1/run_phase1_pipeline.py` 사용 (DXF direct).
- 연구(Research): `graphgen/ai_training/*`는 dataset 준비/학습/평가 scaffold만 제공.
- legacy(`graphgen/ai_phase1_legacy/*`)는 제거하지 않고 다음 용도로 유지:
  - pseudo-label bootstrap 후보 생성,
  - sanity-check baseline,
  - 회귀 비교(reference).

즉, 이번 변경은 운영 추론 경로를 교체하지 않고 학습 실험 시작을 위한 최소 구조를 추가한 것입니다.
