# PR Guide

## 이 PR이 하는 일
- 표준 그래프 이미지 데이터셋 폴더 구조 추가
- 변환 파이프라인 스켈레톤 추가(이미지→previews, placeholder JSON 생성)

## 사용자(데이터) 채우기
- `datasets/standard/images/`에 PNG 이미지들을 넣으시면 됩니다.

## 실행
```bash
python -m graphgen.tools.convert_standard_images
```

PR 제목 추천
	•	Add standard dataset scaffold + converter skeleton
