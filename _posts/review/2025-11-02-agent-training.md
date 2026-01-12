---
title: 멀티모달과 멀티에이전트 조합
category: Review
tags: [NLP, Vision, Multimodal, RL, Agent]
toc: true 
math: true
media_subpath: /assets/posts/sentiment-agent/
---

## Agent의 변화와 정의

> ReAct: Synergizing Reasoning and Acting in Language Models, 2023.
>
> Counterfactual Multi-Agent Policy Gradients, 2024.

Agent는 전통적으로 policy를 가지고 자율적으로 판단하고 행동하는 모델을 말해요. 이는 강화학습 분야에서 오랫동안 사용되던 용어죠. Policy를 가진 네트워크는 학습을 통해 경험을 쌓고, 이를 바탕으로 판단을 내리게 돼요. 최근에는 LLM의 체급을 활용해 LLM 자체를 두뇌로 사용하는 agent가 등장하고 있어요. 그 대표적인 예로 ReAct(Reasoning and Acting) 프레임워크가 있어요. 이러한 형태의 agent는 LLM의 배경지식을 활용하기 때문에 별도의 학습 없이도 동작하거나, 경험을 외부에 저장한 뒤 입력 프롬프트로 사용하는 context learning을 활용해요.

이 글에서는 전통적인 강화학습에 대한 agent 학습을 소개해요. LLM이 높은 범용성과 방대한 지식을 제공하지만, 더 전문적이고 세부적인 특징을 찾기 위해서는 직접 policy를 학습할 필요가 있어요. 또한, LLM을 fine-tune하는 접근법은 학습과 추론 시에 많은 컴퓨팅 파워가 필요하다는 한계가 있어 여전히 전통적인 강화학습 기반 agent가 사용되고 있답니다.

Agent의 성능이 좋아지면서 여러 agent를 이용해 성능을 높이는 연구가 자연스럽게 등장했어요. 그중 대표적인 구조로 Actor-Critic이 있어요. Actor는 직접 행동하며 경험을 학습하고, Critic은 actor의 판단을 평가하며 피드백을 제공해요. 이렇게 여러 agent를 동시에 운용하기 위해 효율적인 협업에 대해 논의가 시작됐어요. 예를 들어, 분산된 환경에서 각 에이전트가 경험을 쌓고, 하나의 중앙화된 학습을 거쳐 가중치를 업데이트하는 구조가 좋은 성능을 보이고 있답니다.

또한, 강화학습과 지도학습의 경계가 점점 모호해지고 있어요. 정답 레이블을 맞추는 문제와 policy를 학습하는 문제를 모두 Loss function에 포함시켜 두 가지 문제를 동시에 학습하는 연구도 확인할 수 있답니다.

## Multi-modal과 multi-agent

> Cooperative Sentiment Agents for Multimodal Sentiment Analysis, 2024.

Multi-modal은 여러 모달리티를 각 encoder가 분석한 뒤 fusion을 통해 하나의 출력을 만드는 구조를 가집니다. 이는 각 agent가 경험을 통해 판단한 뒤 협업을 통해 공통된 과제를 수행하는 모습과 닮아있어요. 이 글에서는 '감정 분석' 문제를 예시로 multi-modal & multi-agent 문제를 어떻게 푸는지 소개할게요.

![Co-SA 구조](co-sa.png)

아이디어는 간단해요. 영상에서 감정을 분석하기 위해 각 모달리티의 특징과 감정 정보를 인코딩한 후, policy를 통해 각 모달리티 정보를 얼마나 반영할지 판단해 최종 출력을 만드는 구조예요. 더 자세히 보면, SAE 단계에서 각 agent는 2개의 인코더를 통해 모달리티의 특징과 감정 정보를 추출해요. SAC 단계에서는 추출된 특징을 하나의 공통된 표현으로 변환해 어떤 정보가 상대적으로 더 중요한지 판단하죠. 마지막으로 downstream task를 풀고 최종 출력을 생성해요.

아이디어는 간단하지만 학습은 그렇지 않아요. Encoder를 어떻게 학습할지, 모달리티 간 관계를 어떻게 학습할지, policy를 어떻게 학습할지 등 고려해야 할 요소가 많아요.

$$L=\alpha_1 L_p + \alpha_2 L_{msd} + \alpha_3 L_{dpsr} + \alpha_4 (L_{actor} + L_{critic})$$

$\alpha_i$는 각 loss term에 대한 가중치이며,

- $L_p$: 감정 레이블에 대한 supervised learning
- $L_{msd}$:
  - Modality 분류 학습 (각 벡터가 어떤 모달리티에서 왔는지)
  - Modality와 Sentiment 간의 이질성(heterogeneity) 학습
  - 원본 입력 복원을 supervised로 학습
- $L_{dpsr}$: 시간에 따른 감정의 변화를 학습
- $L_{actor}$: 각 agent의 policy 학습
- $L_{critic}$: Agent의 평가자 학습

아래에서는 각 Loss term이 어떤 문제를 풀기 위해 사용되었으며 구체적인 식을 정의할게요.

$$L_p$$는 감정 레이블에 대한 predictive loss를 사용해요. 구체적으로 논문에서는 2가지 감정 분류 문제를 해결하기 위해 각 문제에 맞춰 Mean Absolute Loss와 Cross-Entropy Loss를 사용해요.

$$L_m = -\cfrac{1}{3}\sum_{i}^{v,a,t}y_i\log F^m(f_i^m ; \theta^m)$$

$L_m$은 feature vector인 $f^m$이 어떤 모달리티에서 왔는지를 평가해요. $y_v, y_a, y_t$는 각각 이미지, 오디오, 텍스트 레이블이며 순서대로 0, 1, 2로 두고 학습했어요.

$$L_c = -\cfrac{1}{3}\sum_i^{v,a,t}d(f^s_i, f^m_i)$$

$L_c$는 감정 feature $f^s$와 각 모달리티 feature $f^m$의 거리 $d$를 크게 만드는 문제로, 하나의 agent 내에서 감정을 담당하는 인코더와 모달리티를 담당하는 인코더가 겹치지 않는 정보를 학습하도록 유도해요.

$$L_r = \cfrac{1}{3}\sum_i^{v,a,t}d(x_i,\tilde{x}_i)$$

$$\tilde{x}_i = D_i(f_i^s \oplus f_i^m ; \theta^d_i)$$

$L_r$은 추출된 감정과 모달리티 특징을 이용해 원본 입력을 복원하는 문제를 학습해요. 이를 통해 특징 벡터에 입력 데이터의 정보가 얼마나 잘 보존되었는지를 알 수 있어요.

$$L_{dpsr} = \frac{1}{T(T-1)} \sum_{p}^{T} \sum_{q \neq p}^{T} \eta_{pq} \left( \frac{(f_i)_p (f_i)_q}{\|(f_i)_p\| \|(f_i)_q\|} + 1 / 2 \right)$$

$$f_i=W_if_i^s$$

$$\eta_{pq} = T - \|p-q\|$$

$L_{dpsr}$은 프레임 간 감정 벡터의 코사인 유사도를 최소화하도록 학습해요. 이를 통해 프레임 간 다른 감정 특징을 학습하게 강제하며, 시간에 따라 변하는 감정 특징을 추출하도록 도와줘요. 추가로 $\eta_{pq}$를 가중치로 반영해 가까운 프레임 간 차이를 더 많이 반영하도록 조절해요.

$$Q = F_c(\sum_i^{v,a,t}f_i\oplus w_i ; \theta_c)$$

$L_{actor}$는 Q-value를 최대화하는 문제를 해결해요.

$$L_{critic}=Q-\bar{Q}$$

$$\bar{Q}=r+\gamma Q'$$

$$Q'$$는 누적 보상이에요. $$L_{critic}$$은 Temporal-Difference Error 알고리즘을 사용합니다.

### 실험

$$f=(f_v\times w_v)*(f_a\times w_a)*(f_t\times w_t)$$

최종적으로 feature $f$를 이용해 downstream task를 풀어요. Visual encoder는 Facet, Acoustic encoder는 COVAREP, text encoder는 BERT를 사용했어요.

### 결과

![IEMOCAP 평가](result.png)

학습한 데이터 중 IEMCOCAP를 평가한 표예요. 본 논문에서 제안한 Co-SA가 모든 항목에서 좋은 성능을 보였어요.

![Feature space](features.png)

Feature가 학습된 공간을 보면 각 모달리티 별로 잘 분리되어 학습된 것을 확인할 수 있어요. 같은 모달리티 내에서도 모달리티 특징과 감정 특징을 따로 학습했답니다.

![dpsr loss 효과](dpsr.png)

$L_{dpsr}$은 프레임 간 다른 감정 정보를 학습하게 해 시간에 따른 감정 변화를 파악해요. 그래프는 프레임 간 유사도가 작게 학습된 것을 보여주고 있어요. 이를 통해 프레임의 특징적인 정보를 추출했다고 해석할 수 있어요.

---

이 글에서 소개한 논문은 멀티모달 문제를 멀티 에이전트 형식으로 풀었어요. 무엇보다 Loss function을 정교하게 설계해서 각 모달리티 별로, 에이전트 별로, 프레임 별로 다른 특징을 학습하도록 했다는 점이 흥미롭네요.
