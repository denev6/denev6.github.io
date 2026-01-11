---
title: Agent에게 오답노트를 시켜봤더니
tags: [AI, Agent, LLM]
category: Study
toc: true
media_subpath: /assets/posts/agent-memory/
---

![](ai-robot-aha.jpg)

## Fine-tuning은 가성비가 떨어진다

> A Survey on In-context Learning, 2024.

LLM은 최근 다양한 분야에서 활용되고 있어요. 특정 과제의 성능을 높이기 위해 지도학습과 강화학습을 결합한 fine-tuning 기법이 사용되고 있지만, 이 방법은 가성비가 떨어진다고 해요. 계산 비용이 매우 높고, LoRA처럼 효율적인 학습 방법이 제안되었지만 여전히 LLM을 돌리는 데는 많은 GPU 연산이 필요해요. 학습된 모델은 특정 작업에만 최적화되어 범용성을 포기해야 하며, 양질의 데이터도 다량으로 필요해요. 이러한 이유로 fine-tuning은 까다로운 방법이에요.

그래서 fine-tuning 없이 성능을 높이는 방법이 연구되어 왔어요. 초기에는 *Prompt Engineering*이라는 표현이 사용되었고, 이후 *Context Engineering*이라는 개념이 등장했어요. Context Engineering은 입력 프롬프트를 상황에 맞게 동적으로 조절해 LLM의 능력을 최대한 활용하는 기법이에요. 최근에는 *Context Learning*도 등장했는데, 이는 context에 제공된 예시를 통해 reasoning하는 방법이에요. 파라미터를 업데이트하지 않지만 few-shot과 같은 직관을 공유해요.

## 오답노트를 쓰게 하다

LLM을 사용하다 보면 비슷한 실수가 반복되곤 해요. 연구자들은 이 점에 주목했어요. 과거의 실수를 기록하고, 비슷한 상황에서 참고할 수 있게 하는 것이죠. 이 글에서 소개할 `Reasoning Bank`와 `Training-Free GRPO`는 각각 Google과 Tencent에서 제시한 오답노트 작성법이에요. `ACE`라는 기법도 비슷한 맥락의 논문이지만 여기서는 다루지 않을게요.

> Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models.

### Reasoning Bank

> ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory, 2025.

Agent memory를 통해 과거 경험을 재활용하려는 연구가 있었지만 두 가지 문제가 있었어요.

1. 기록만 쌓아두고 추상적이고 일반화된 추론 전략을 뽑아내지 못했어요.
2. 성공한 기록만 강조하고 실패에서 얻을 교훈을 반영하지 못했어요.

Google에서 제시한 ReasoningBank는 추상화된 전략을 찾아 기록하며 성공과 실패에서 얻은 주요 전략을 함께 기억해요.

![Reasoning Bank](reasoningbank.png)

Agent는 가장 유사한 k개의 기억을 가져와 프롬프트와 함께 입력해요. 이 프롬프트를 사용해 결정을 내리고, 결과에서 새로운 지식을 찾아 title, description, content 형식으로 정리해요. 이렇게 정리된 memory item은 새로운 기억으로 추가돼요.

- title: 주요 전략
- description: 한 줄 요약
- content: 경험으로부터 배운 전략과 지식

논문에서 강조하는 것 중 하나가 Memory-Aware Test-Time Scaling (MATTS)예요. MATTS는 inference time에 정보를 추가로 학습하는 기법으로, Parallel Scaling과 Sequential Scaling 두 가지 방법을 제안해요. Parallel Scaling은 여러 경로의 출력을 생성하고 공통된 reasoning 패턴만 취하는 방식이고, Sequential Scaling은 self-refinement를 통해 reasoning을 다듬어가는 방식이에요.

실험 결과, ReasoningBank를 사용하면 성능이 향상되었고 MATTS를 적용하면 더 높은 성능을 보였어요. Sequential Scaling은 단기 상황에 강점을 보였고, Parallel Scaling은 장기적이고 일반화된 상황에 강점을 보였어요. 또한 ReasoningBank를 적용하면 과제를 수행하는 데 걸리는 단계가 줄어들어 agent가 효율적인 reasoning 경로를 찾았음을 나타내요.

### Training-Free GRPO

> Training-Free Group Relative Policy Optimization, 2025.

Tencent는 과거 경험이 학습된 token prior 역할을 해 파라미터 업데이트와 유사한 효과를 낸다고 주장해요. 기존 강화학습에서 사용하던 Group Relative Policy Optimization (GRPO) 개념을 가져와 파라미터 학습 없이 모델 성능을 개선했어요. 강화학습에서의 GRPO는 여러 출력을 생성하고 각 출력이 독립적인 reward를 받는 방식이에요. Training-Free GRPO는 이 아이디어를 가져와 각 출력의 성공/실패를 판단해 공통 원인을 요약하고 저장해요. Google의 Parallel Scaling과 같은 이야기를 하고 있어요.

![GRPO](grpo.png)

GRPO는 reward를 이용해 그룹 내 상대적 advantage를 계산하고 PPO로 파라미터를 업데이트해요. Training-Free GRPO는 이 과정 대신 생성된 지식을 추가해 optimization을 수행해요. Add, Delete, Modify, Keep 연산 중 하나를 선택해 기억을 업데이트하는 방식이에요. 이 아이디어는 이전 *Long-term Memory*라는 이름으로 제안되었던 방법론을 응용한 것이에요.

실험 결과, Training-Free GRPO는 뛰어난 성능 향상을 보여줬어요. 올바른 reasoning을 유도했을 뿐만 아니라 agent가 적절한 tool 사용도 배우도록 했어요. 출력 그룹을 활용해 여러 경로를 생성하는 것이 하나의 출력만 생성하는 것보다 좋은 성능을 보였어요.

fine-tuning을 사용한다면 학습에 $10,000 정도 들었을 문제를 $18 정도로 학습할 수 있었어요. 추론 시 fine-tuning은 시간당 $0.005가 들지만, Training-Free GRPO는 $0.02가 들죠. 하지만 GPU 서버 관리 없이 분산된 LLM 서비스 인프라 활용이 가능하므로 real-world application에 더 적합할 수 있다고 주장해요.

### 한계점

Agent의 경험에서 중요한 지식을 찾아 context에 주입하는 방법을 알아봤어요. 하지만 이 방법도 결국 Base LLM의 추론 능력이 뒷받침되어야 가능해요. 논문에서 사용한 LLM 모델도 Gemini-2.5-Pro, Claude-3.7-Sonnet, DeepSeek-V3.1-Terminus 등 기본 체급이 있는 모델들이에요. reasoning 능력이 낮은 작은 모델을 사용한다면 context 관리를 잘해도 결과를 보장할 수 없어요. Gemini-2.5-flash, Qwen3-32B-Instruct 수준의 모델로 실험했더니 성능 향상이 있었지만 반대로 크게 떨어지는 상황도 관찰되었어요.

또한 agent 출력의 성공/실패 판단 시 ground truth 없이 LLM-as-a-judge를 사용하는데 판단이 모호한 경우 noise로 작용할 수 있어요. 실험에서는 이러한 noise에 robust하다는 사실이 확인되었지만 여전히 발전된 judge 적용이 과제로 남아있어요.

---

**참고문헌**

- ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory, 2025.
- Training-Free Group Relative Policy Optimization, 2025.
- Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models, 2025.
- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory, 2025.
- LoRA: Low-Rank Adaptation of Large Language Models, 2021.
- A Survey on In-context Learning, 2024.
