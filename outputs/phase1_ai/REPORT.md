# Phase1 Report — DXF Direct Path

- source dxf: `examples/Drawing1.dxf`
- nodes: 41
- edges: 47 (100% directed)
- stations: 10

## Produced files
- `outputs/phase1_ai/graph.json` — sim_core 입력용 그래프
- `outputs/phase1_ai/preview.svg` — SVG preview (DXF 기하 기반)
- `outputs/phase1_ai/directed_graph.png` — matplotlib PNG
- `outputs/phase1_ai/REPORT.md` — 이 리포트

## Method
DXF → parse → outer-loop CW → direction propagation → export
(이미지 경유 역추출 제거됨, see issue #12)