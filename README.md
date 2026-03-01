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
