---
title: Generalist Robot Policy 살펴보기
category: Review
tags: [Vision, Multimodal]
toc: true 
math: true
media_subpath: /assets/posts/general-vla/
---

범용 지능인 LLM의 성능이 크게 발전하면서 이 능력을 물리 세계로 옮기려는 시도가 활발해지고 있어요. 본 글에서는 물리 세계를 이해하고 상호작용할 수 있는 Physical AI의 동향을 정리하고 각 논문에서 소개한 주요 아이디어를 알아보려고 해요. 논문 선정은 [앵지유니버스](https://youtu.be/8V2a8Ty5-yk?si=xChZzpGCVUr9MENQ)님의 영상을 참고했어요.

## Generalist VLA의 등장

![octo](octo.webp)

기존의 로봇은 센서, 동작 범위 등 요소가 바뀌면 처음부터 새로 학습해야 하는 문제가 있었어요. 이 때문에 다양한 downstream task에 적용할 수 있는 범용 모델이 필요했고 Octo가 등장했죠. Octo는 transformer 기반의 정책을 사용해요. 입력 tokenizer는 language instruction, goal, observation sequence를 토큰으로 변환해요. 이때 자연어는 t5-base, 이미지는 shallow convolutional stack을 사용해요. 생성된 토큰은 positional encoding을 적용한 상태로 transformer backbone을 거쳐 임베딩으로 변환되고, 최종적으로 readout head를 통해 출력을 생성해요. Readout의 출력은 BERT의 \[CLS\] 토큰과 같이 정보를 압축해 놓은 벡터의 역할을 해요. 따라서 최종적으로 가벼운 action head를 거치면서 action chunk를 예측해요. Action head는 conditional diffusion decoding을 통해 출력을 생성하기 때문에 standard DDPM objective로 학습해요.

![OpenVLA](openvla.webp)

OpenVLA는 Octo와 마찬가지로 범용 오픈소스 VLA를 목표로 등장했으며, 성능이 더 뛰어나고 fine-tuning이 쉽게 설계되었어요. 저자는 행동 예측 문제를 'vision-language' task로 정의했어요. 이를 위해 연속된 행동을 bin 단위로 나누어 이산화된 토큰으로 정의했어요. 그리고 LLM backbone의 출력 토큰 중 자주 사용하지 않는 토큰을 256개의 행동 토큰으로 덮어쓰는 방식으로 행동 공간이 LLM의 토큰 공간 안에 통합되도록 설계했어요. 따라서 학습도 standard next-token objective와 동일하게 cross-entropy loss를 계산해요. 이러한 설계 덕분에 LoRA나 4-bit quantization과 같이 LLM에서 사용하던 테크닉을 적용할 수 있다는 장점을 가져요. 참고로 이미지는 visual encoder + projector를 통해 LLM의 벡터 공간으로 매핑되며, SigLIP-DINOv2가 spatial reasoning 성능을 향상시켜, 널리 사용되는 CLIP + SigLIP 조합보다 높은 성능을 보였다고 해요.

## 생성 모델로 연속 행동 예측

![pi0](pi0.webp)

$\pi_0$는 대량의 데이터를 올바른 구조와 방법으로 학습하는 것이 중요하다고 주장해요. 먼저 모델 구조의 관점에서는 [Conditional Flow Matching](https://youtu.be/wFikV0Y7NIk?si=D8m17ODr5lGhS18S)을 이용한 행동 생성을 제안하며, action chunking을 학습하기 위해 flow matching loss를 사용해요. 또한 학습 전략에 대해서도 설명해요. 초반 pre-train 단계에서는 넓은 범위의 일반화 지식을 학습하도록 하고, post-training 단계에서는 정교하고 복잡한 작업을 학습하도록 해요. 이 방식을 통해 일부 문제에서 성능을 최대 2배 높였어요.

![cogact](cogact.webp)

CogACT는 diffusion transformer를 이용해 action을 생성해요. 특별한 점은 자연스러운 연속 행동 예측을 위해 adaptive ensemble strategy를 도입했다는 점이에요.

$$\hat{a}_t = \sum_{k=0}^{K} w_k^{\text{ada}} \cdot a_t | o_{t-k}$$

$$w_k^{\text{ada}} = \exp(\alpha \cdot < a_t | o_t, a_t | o_{t-k} >)$$

$a_t \| o_{t}$는 관찰된 상태 $o_{t}$에 대해 예측한 행동 $a_t$를 의미해요. $< \cdot , \cdot >$은 두 값 사이의 유사도를 뜻해요. 즉, 현재 $a_t$는 과거 $a_{1, ... t-1}$의 영향을 받으며, 과거 행동 중 유사도가 낮은 행동의 가중치를 지수적으로 감소시켜 과거 시퀀스의 정보가 현재 $o_t$에 자연스럽게 반영되도록 해요.

## Embodied Reasoning

![Gemini-robotics 구조](gemini-robotics.webp)

Google은 gemini를 이용해 gemini-robotics를 설계했어요. gemini의 풍부한 지식 덕분에 zero-shot만으로도 뛰어난 embodied reasoning 성능을 보였어요. Embodied reasoning이란 2D/3D object detection, 2D pointing, multi-view correspondence 등 과제를 수행하는 능력을 말해요. 여기에 CoT나 few-shot을 적용하면 성능이 향상되었어요. CoT는 단순히 *"Reason step by step about the answer, and show your work, for each step. Only after that, proceed to the final answer"*를 지시문 끝에 붙이는 걸 말해요. Few-shot은 시연 예시를 inference time의 프롬프트에 추가하는 in-context example을 말하며, 동작뿐만 아니라 동작에 대한 reasoning을 자연어로 서술해 함께 포함시켜요. 두 방식 모두 성능 향상을 보였으며, 특히 few-shot은 zero-shot일 때 성공하지 못한 과제를 성공하기도 했어요. 더 나아가 inference latency를 줄이기 위해 모델을 두 모듈로 분리했어요. Gemini를 기반으로 한 VLA backbone은 cloud에서 작동시켜 지연 시간을 160ms 미만으로 줄였으며, Action decoder는 로봇에 위치해요. 이 구조를 통해 end-to-end 작업에 약 250ms의 지연이 발생해 실시간 추론이 가능한 수준의 성능을 달성했어요.

![Gemini-robotics 성능](gemini-success.webp)

Gemini-robotics의 가장 큰 장점은 범용 지식을 학습하고 있다는 점이에요. 이 덕분에 fine-tuning을 거쳤을 때 평균 과제 성공률 79%를 달성했어요. 일부 과제에서 baseline 모델들은 0점을 달성했으나, gemini-robotics는 과제에 따라 적게는 45%부터 높게는 100%의 성공률을 보이며 다른 모델 대비 눈에 띄게 높은 성공률을 보였어요. 강력한 일반화 성능 덕분에 다른 embodiment로의 전이도 가능했어요. 예를 들어, ALOHA 2에서 수집한 데이터로 학습한 모델을 bi-arm Franka에 전이(fine-tune)했을 때 평균 과제 성공률 63%를 달성했어요.

---

**Reference**

- Team, Octo Model, et al. "Octo: An open-source generalist robot policy." arXiv preprint arXiv:2405.12213 (2024).
- Kim, Moo Jin, et al. "Openvla: An open-source vision-language-action model." arXiv preprint arXiv:2406.09246 (2024).
- Black, Kevin, et al. "π0: A vision-language-action flow model for general robot control, 2024a." arXiv preprint arXiv:2410.24164 (2024).
- Li, Qixiu, et al. "Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation." arXiv preprint arXiv:2411.19650 (2024).
- Team, Gemini Robotics, et al. "Gemini robotics: Bringing ai into the physical world." arXiv preprint arXiv:2503.20020 (2025).
