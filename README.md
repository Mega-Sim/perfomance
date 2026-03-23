# OHT Simulator – Graph Engine (GitHub Ready)

DXF 레이아웃에서 OHT 단방향 그래프를 생성하는 **Graph Engine**입니다.

## 핵심 규칙
- **외곽 루프(CW) 고정**
- **분기/엣지 방향 전파**
- **모든 Station 경로 가능(강연결)**
- **직선 스켈레톤 완성 후 곡선(ARC) 복원**
- **CAD에 없는 엣지 추가 금지**

## 설치
Python 3.9+

```bash
pip install -r requirements.txt
```

## 실행

예시 도면으로 실행:

```bash
python src/build_graph.py examples/Drawing1.dxf outputs/out.png outputs/out.json
```

결과:
- `outputs/out.png`: 방향 표시 그래프
- `outputs/out.json`: 그래프 덤프(노드/엣지/기하 포함)

## 현재 범위
- Layout/Graph Builder 까지 구현 완료
- Event Engine / Traffic / Scheduler / Multi-OHT는 아직 포함되지 않음

## Phase1: Graph extraction (DXF direct path)

DXF에서 직접 그래프를 추출합니다 (이미지 경유 없음).

```bash
# 방법 1: build_graph.py 직접 실행
python src/build_graph.py examples/Drawing1.dxf outputs/out.png outputs/out.json

# 방법 2: pipeline runner 사용
python -m graphgen.ai_phase1.run_phase1_pipeline --dxf examples/Drawing1.dxf
```

결과:
- `graph.json`: sim_core 입력용 방향 그래프 (100% directed)
- `preview.svg`: DXF 기하 기반 벡터 preview
- `directed_graph.png`: matplotlib 렌더링

Note: 기존 이미지 경유 역추출 코드(색상 세그멘테이션 + 스켈레톤화)는
`graphgen/ai_phase1_legacy/`로 이동되었습니다. 자세한 사유는 issue #12 참고.

## Phase2: Direction refinement (heuristic + PPO baseline)

Phase2는 `solve()` 이후에 기존 방향 할당을 **keep/flip** 방식으로 보정합니다.

- 기본 모드: `heuristic` (기존 deterministic fallback)
- 학습 모드: `ppo` (Stable-Baselines3 PPO 정책 로드/추론)
- 보상 기준: dead-end / station reachability / nonholonomic violation 기반 점수 개선

### PPO 학습

```bash
python -m graphgen.ai_phase2.train_phase2_rl \
  --dxf examples/Drawing1.dxf \
  --total_timesteps 20000 \
  --model_out outputs/phase2/ppo_drawing1 \
  --seed 42
```

### Phase1 파이프라인에서 Phase2 PPO 추론 사용

```bash
python -m graphgen.ai_phase1.run_phase1_pipeline \
  --dxf examples/Drawing1.dxf \
  --use_phase2 \
  --phase2_mode ppo \
  --phase2_model outputs/phase2/ppo_drawing1.zip
```

> 현재 PPO 경로는 첫 번째 learned-policy baseline(MLP + fixed vector observation)이며, 최종 아키텍처가 아닙니다.
