---
# the default layout is 'page', 'tag' and 'archives' are also available.
icon: fas fa-info-circle
order: 1
---

## 박성진

Sung-jin Park

- sungjin.code@gmail.com
- Github: [denev6](https://github.com/denev6)
- Seoul, South Korea.

## 소개

- 인공지능을 전공하고 있는 학부생이며, Agent와 Reasoning에 관심이 많습니다.
- 새로운 기술을 빠르게 학습하고, 이를 통해 실질적인 문제를 해결하는 과정에 흥미를 느낍니다.
- 학습한 내용을 정리하고 공유하여 팀의 지속적인 성장을 이끌고자 합니다.

## 학력 및 경력

- 성균관대학교 LAMDA Lab 학부연구생 '25.02 ~ 12
- 성균관대학교 창업프로그램(SPARK) 15기: '25.05 ~ 09
- 성균관대학교 인공지능융합학과: '21.03 ~ (GPA: 4.41)

## 기술 스택

- Language: `Python`, `Go`
- Data: `Numpy`, `OpenCV`, `Pandas`, `Scikit-learn`, `Matplotlib`
- Machine-learning: `Pytorch`, `Huggingface`, `Unsloth`, `Hydra`, `LangGraph`, `Mediapipe`
- MLOps: `Docker`, `vLLM` <!--, `TensorRT`, `Kubernetes`, `Kubeflow`, `Prometheus`, `Kafka`-->
- Backend: `FastAPI` <!--`Django`, `Gin`, `gRPC`-->
- DB: `RDBMS`, `VectorDB`
- Tools: `Git`, `Ubuntu`, `Conda`, `uv`, `WandB`, `Jira`

## 프로젝트

### 💬 멀티-에이전트를 활용한 의료 상담 챗봇

- '25.06 ~ 12 (팀)
- 말기 암 환자 대상 챗봇 개발을 위해 LangChain, FastAPI, SvelteKit을 활용한 풀스택 개발을 담당했습니다.
- 멀티-에이전트 구조를 도입했으며, 상담의 전문성·공감성 등 7개 평가지표에 대해 단일 LLM 대비 평균 9.4%의 성능 향상을 달성했습니다.
- Agent-router의 프롬프트를 개선했으며, 룰 기반 시스템 대비 정확도를 53.4% 향상시켰습니다.
- 실제 의료 현장의 니즈를 반영하기 위해 삼성서울병원 의료진과의 협업을 통해 시스템을 기획·평가했습니다.
- [더보기](/posts/eirene)

### 💻 FaceMouse: 얼굴 인식을 활용한 접근성 개선

- '22.04 ~ 11 (팀), '25. 03 ~ 05 (개인)
- 상지 장애인의 웹 접근성을 개선하기 위해 얼굴 주시 방향으로 마우스 커서를 제어하는 소프트웨어입니다.
- MediaPipe와 OpenCV로 머리 회전을 이용한 커서 제어와 눈 깜박임을 통한 클릭을 구현했습니다.
- 표준 웹캠과 CPU-only 환경에서 실시간 동작하도록 파이프라인을 최적화하여 실행 시간을 최대 12배 단축했으며 CPU 사용률을 최대 40.7%p 개선했습니다.
- 정확도와 실사용 가능성을 검증하기 위해 ±37.7°의 머리 회전 범위에서 포인터 추적 성공률 100%를 기록하며 마우스 대체 입력 장치로서의 기능적 유효성을 실험적으로 입증했습니다.
- 반복적인 사용자 인터뷰와 함께 정량적인 모니터링 수행하여 CPU 오버헤드로 인한 문제를 해결하고 사용자 경험을 개선했습니다.
- [더보기](/posts/facemouse)

### 기타

- 😎 Tiny Reasoning을 이용한 Multi-modal 분류 모델 학습: [더보기](/posts/multimodal-trm)
- 🎙️ 실시간 음성인식 파이프라인 최적화 및 LLM 기반 모바일 GUI 제어: [더보기](/posts/speak2ui)

## 논문 및 특허

- "말기 암 환자의 임종 의사결정 지원을 위한 멀티에이전트 시스템", 한국정보통신학회, 2025. (게재 예정)
- "Speak2UI: 모바일 접근성을 위한 음성 보조 기술", 한국정보통신학회, 2025. (게재 예정)
- "얼굴 인식과 마우스 제어 기반의 비접촉식 입력 방법 및 상기 방법을 수행하는 장치", [1020230063378](https://doi.org/10.8080/1020230063378) ('25.11.04).
- 얼굴 인식과 Pyautogui 마우스 제어 기반의 비접촉식 입력 기법, 한국정보통신학회, 2022. [10.6109/jkiice.2022.26.9.1279](https://koreascience.or.kr/article/JAKO202228049092231.page?&lang=ko)

## 수상 내역

- **2025 Bias-A-Thon**(Bias 대응 챌린지 Track 2): 3위 ('25.06) [더보기](/posts/dacon-bias)
- 월간 DACON **발화자의 감정인식 AI 경진대회**: 2위 ('22.12) [더보기](/posts/dacon-roberta)
- 성균관대학교 **Co-Deep Learning**: 우수상 ('22.08)
