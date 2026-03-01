## Pipeline
DXF → parse (LINE/ARC/TEXT) → outer loop CW → direction propagation → graph.json + preview.svg

### Phase1 (현재)

DXF ──→ build_graph.py ──→ graph.json (sim_core 입력)
├─ preview.svg (DXF 기하 벡터 렌더)
└─ directed_graph.png (matplotlib)

### Phase1 이전 (deprecated, see issue #12)

DXF → matplotlib PNG → 색상 세그멘테이션 → 스켈레톤화 → 그래프 역추출
(정보 손실로 인해 제거됨, 코드는 graphgen/ai_phase1_legacy/ 에 보관)
