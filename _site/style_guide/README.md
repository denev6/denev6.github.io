# Blog Style Guide

- Projects & Playground:
  - Icons: [doodle-icons](www.svgrepo.com/collection/doodle-icons/)
  - Fonts: Architects Daughter, Kyobo Handwriting 2021
- Study:
  - Icons: Google Material Icons
  - Font: Pretendard

```sh
$ python style_guide/main.py
```

## 토스 가이드

1. **가치를 먼저 제공하기**: 기능 중심의 설명보다는 독자가 얻을 수 있는 가치를 먼저 설명하는 것이 좋아요. 독자가 문서를 읽고 어떤 문제를 해결할 수 있는지, 어떤 긍정적인 변화를 기대할 수 있는지 먼저 전달해야 해요.
2. **효과적인 제목 쓰기**: 제목은 문서의 핵심 내용을 간결하고 명확하게 전달해야 합니다. 독자가 문서를 스캔하면서 빠르게 내용을 파악할 수 있도록, 명확하고 검색하기 쉬운 제목을 작성하는 것이 중요합니다.
3. **예측 가능하게 하기**: 문서는 독자가 쉽게 탐색하고 필요한 정보를 빠르게 찾을 수 있도록 예측 가능해야 합니다. 예측 가능한 문서 구조는 독자가 내용을 이해하는 데 드는 인지 부담을 줄이고, 문서를 효율적으로 활용할 수 있도록 돕습니다. 예측 가능한 문서는 일관된 용어와 표현을 사용하고, 설명의 흐름이 자연스럽게 연결되며, 새로운 정보가 논리적인 순서로 배치됩니다.

### 체크리스트

- 도구나 기술, 시스템을 행동의 주체로 사용하지 않아요
- 능동형으로 표현해요
- 문장을 짧고 간결하게 유지하세요
- 메타 담화를 최소화하세요
- 명사 대신 동사를 사용하세요
- 모호한 표현 대신 명확한 표현을 사용하세요
- 문장에서 누가, 무엇을, 어디에, 어떻게 하는지 써주세요
- 실제 동작을 이해할 수 있게 쓰세요
- 구체적인 수치나 기준을 제시하세요
- 참조되는 내용의 맥락을 설명하세요
- 불필요한 한자어를 제거하세요
- 번역체 표현을 자연스럽게 수정하세요
- 공식 기술 용어를 따르세요
- 같은 개념을 여러 방식으로 표현하지 마세요
- 약어는 먼저 풀어쓴 후 사용하세요
- 외래어 표기는 사용 빈도를 고려하세요

## 프롬프트

프롬프트 아래에 마크다운 텍스트를 첨부합니다.

````text
<Document>
```markdown
실제 문서 내용
```
</Document>
````

### 1. 톤 가이드

- system: [tone_sys.md](./tone_sys.md)
- user: [tone_user.md](./tone_user.md)

### 2. 검토 가이드

- system: [eval_sys.md](./eval_sys.md)
- user: [eval_user.md](./eval_user.md)

## 참고자료

1. [당근 테그 블로그](https://medium.com/daangn)
2. [토스 테크니컬 라이팅 가이드](https://technical-writing.dev/)
3. [네이버 CLOVA 블로그](https://clova.ai/tech-blog)
4. [Apple Developer 유튜브](https://www.youtube.com/@AppleDeveloper)
