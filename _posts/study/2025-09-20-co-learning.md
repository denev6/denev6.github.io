---
title: Co-learning을 활용한 멀티모달 학습
tags: [AI, CV, NLP]
category: Study
toc: true
img_path: /assets/posts/co-learning/
---

## Co-learning

Multi-modal learning에서 co-learning은 모달리티를 학습하기 위해 다른 모달리티를 활용하는 방법을 말한다. 예를 들어, "강아지"라는 텍스트의 특성을 파악하기 위해 강아지 이미지를 함께 학습하는 식이다.

본 글은 대표적인 두 모델 `ViLBERT`와 `CLIP`을 살펴본다.

## ViLBERT

![ViLBERT](vilbert-overview.png)

기본적으로 `BERT` 구조를 사용해 특징을 추출한다. 각 모달리티의 데이터를 embed한 뒤 `co-attention`을 통해 모달리티 간 정보를 상호 학습한다.

![co-attention](co-attention.png)

`co-attention`은 자신의 모달리티를 Query로, 다른 모달리티를 Key-Value로 사용한다. 예를 들어 Visual block은 강아지 이미지를 Query로 사용한다. Key와 Value로  *"A pile of sleeping puppies"*라는 문장을 사용한다. 그럼 이미지 query와 문장 key의 관계를 파악해 서로 어떤 정보가 연결되어 있는지 학습한다. 마지막으로 value로 들어온 문장 *"A pile of sleeping puppies"*의 각 토큰이 이미지의 어떤 부분과 얼마나 관련 있는지 매핑한다.

![attention](attention-result.png)

### 언어 모델 학습

BERT와 마찬가지로 단어를 토큰 단위로 임베딩하며, `<SEP>`, `<MASK>`, `<CLS>`와 같은 특수 토큰을 사용한다. Attention block 전에 positional encoding을 통해 위치 정보를 주입한다. 단순히 단어 토큰 간 관계를 파악하는 것이 목적이지만, 이 경우 명확한 ground-truth 레이블을 매기기 힘들다. 따라서 BERT에서 했던 것과 마찬가지로 2개의 과제를 만들어 학습에 사용한다.

![language learning](language-learning.png)

Masked Language Modeling은 랜덤하게 15% 정도의 토큰을 마스킹한다. 이 중 80%는 `<MASK>`로 대체되고, 10%는 랜덤한 단어로 대체되며, 10%는 그대로 유지한다. 마지막에 유지되는 10%의 토큰은 원본과 동일하지만 예측 대상에 포함된다. 모델은 문장 중간에 비어있는 단어를 예측하거나 올바른지 확인하면서 문맥 이해 능력을 학습한다. 구체적으로, 마지막 hidden layer의 출력을 정답 레이블과 비교해 가중치를 학습한다.

Next Sentence Prediction 문제는 두 문장을 이어 붙이고 자연스러운 문장이 이어졌는지 판단한다. 예를 들어, "\<CLS\> 노트북이 고장났다 \<SEP\> 급하게 수리점에 갔다 \<SEP\>"를 입력으로 준 뒤, `<CLS>` 토큰을 Linear layer로 보내 두 문장이 관련된 정도를 계산한다. 이를 정답 레이블과 비교해 가중치를 학습한다.

### 시각 정보 학습

![image embeddings](image-embed.png)

먼저 Faster R-CNN을 이용해 이미지 내의 object bounding box를 추출하고 이 정보를 이미지 feature vector와 spatial(positional) encoding 정보로 사용한다. Spatial encoding은 bounding box의 top-left 좌표(2개), bottom-right 좌표(2개), 객체 영역이 전체에서 차지하는 비율(1개)로 구성된 5D 벡터를 가진다. Faster R-CNN의 내부에서 추출된 feature vector와 spatial encoding을 합쳐 최종적으로 image embedding을 출력한다. 참고로 이미지의 시작은 `<IMG>` 토큰을 사용한다.

Multimodal learning도 마찬가지로 masked learning과 alignment prediction 문제를 풀며 학습한다.

![visual learning](visual-learning.png)

Masked Learning은 마스킹할 이미지 토큰 15%를 정하고, 이 중 90%은 0(zero)으로 처리하고 10%는 그대로 둔다. 이렇게 예측된 마스킹 이미지 토큰의 확률 분포는 Faster R-CNN의 분포를 정답으로 간주하고  KL divergence로 학습한다. 예를 들어, ViLBERT가 1번 이미지 패치를 { 80% 강아지, 12% 토끼, … }로 예측했고, Faster R-CNN이 { 93% 강아지, 3% 늑대 … }로 예측했다면 Faster R-CNN의 확률 분포에 가까워지도록 ViLBERT의 가중치를 조정한다.

Alignment Prediction은 이미지의 특징을 가진 `<IMG>` 토큰과 문장의 특징을 가진 `<CLS>` 토큰을 Linear layer에 넣고 이미지와 문장이 관련 있는지 판단한다. 언어 모델을 학습할 때 두 문장이 관련 있는지 예측했던 것처럼 이번에는 image-caption의 관계를 분석하는 과제를 통해 두 모달리티 간 관계를 학습한다.

### 모델 응용

![vilbert examples](vilbert-task.png)

ViLBERT는 두 모달리티의 관계를 학습하고, 이미지와 텍스트를 모두 이해하는 가중치를 학습하는 것이 목표다. 이렇게 학습된 베이스 모델은 여러 downstream task를 풀기 위해 fine-tune할 수 있다.
