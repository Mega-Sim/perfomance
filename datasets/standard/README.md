# Standard Layout Dataset (Spec v1.1)

이 디렉터리는 Mr. SIK 표준 표기를 따르는 이미지 데이터셋 기준정보를 담습니다.

## 표준 표기
- 트랙: 초록색(굵은 선)
- 배경: 하얀색(단색)
- 노드: 빨간색 원형 점 (required)
- 스테이션: 파란색 사각형 점 (optional)
- 방향: 트랙 바깥의 초록 화살표 (required; 트랙과 겹치지 않게)

## Validator 실행
```bash
pip install -r graphgen/requirements.txt
python -m graphgen.tools.validate_standard_dataset
```

`datasets/standard/images/`는 이후 사용자가 채울 예정입니다.
