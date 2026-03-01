# graphgen

표준 레이아웃 spec 로드와 validator 실행 예시:

```bash
python -c "from graphgen.spec import load_standard_spec; print(load_standard_spec()['spec_version'])"
python -m graphgen.tools.validate_standard_dataset
```
