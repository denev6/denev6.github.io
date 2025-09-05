---
title: FaceMouse - Empowering Web Accessibility with a Lightweight and Open-Source Interaction System
tags: [AI, CV, Python]
category: Projects 
toc: true
img_path: /assets/posts/facemouse/
---

<video src="https://github.com/user-attachments/assets/8b51e391-7c63-49dc-920b-28960477943e" width="100%" controls muted></video>

- v1: 2022-07
- **v2: 2025-05**

본 연구는 상지 장애인의 디지털 접근성을 개선하기 위해 세 가지 기여를 통한 새로운 비접촉 입력 시스템을 제안한다.

- 첫째, 마우스를 대체하는 새로운 입력 기법을 개발했다. Face Mesh 모델을 활용하여 머리 움직임을 통한 정밀한 커서 제어를 구현하고, 눈 깜박임 감지로 클릭 명령을 처리하는 직관적 상호작용 방식을 제시했다. 성능 평가 결과 저전력 CPU 환경에서 100% 성공률과 최대 37.7도의 각도 측정 범위를 달성하여 실용적인 마우스 대체 솔루션임을 입증했다.
- 둘째, 저비용·저사양 환경에서 사용 가능한 실용적 시스템을 구축했다. 전용 하드웨어나 GPU 없이 표준 웹캠과 일반 CPU만으로 12.4% CPU 사용률, 32.93MiB 메모리 사용량, 1.12초 실행 시간의 최적화된 성능을 달성했다. 이는 상용 보조 기술과 달리 일반 사용자의 기존 컴퓨팅 환경에서 즉시 활용 가능한 경제적 접근성을 제공한다.
- 셋째, 오픈소스 공개를 통해 기술 진입 장벽을 해소했다. 완전한 소스 코드 공개로 구매 비용을 제거하고, 사용자 맞춤 설정과 커뮤니티 기반 지속적 개선을 가능하게 했다. 장애가 있는 사용자와 없는 사용자를 대상으로 한 반복적 사용자 테스트를 통해 시스템의 사용성을 검증하고 피드백을 반영한 개선을 수행했다.

> *팔레트는 주인공이 되기보단, 주인공이 가장 잘 빛날 수 있도록 바탕을 만들어주는 도구입니다. 자신이 주인공은 아니지만, 물감 하나하나가 아름답게 빛날 수 있도록 돕는 존재죠. 저희 소프트웨어는 누군가를 대체하거나 주인공이 되려는 것이 아닙니다. 사용자 한 사람 한 사람이 주인공이 되어 빛날 수 있도 록 조용히 곁에서 돕는 것이 저희의 목표입니다. 화려하지는 않지만, 기술을 통해 더 많은 사람이 자유롭고 편안하게 일상을 살아갈 수 있도록 보이지 않는 곳에서 함께하고 싶습니다.* - Team Palette

## 서론

현대 디지털 사회에서 컴퓨터 인터페이스는 교육, 의료, 금융 서비스 등 사회 전반에 널리 활용되고 있다. 특히 웹 기반 플랫폼의 확산으로 인해 디지털 접근성은 사회 참여의 필수 조건이 되었으나, 상지 장애를 가진 사용자는 표준 마우스 사용의 어려움으로 인해 디지털 소외를 경험하고 있다. 세계보건기구(WHO)에 따르면 전 세계 인구의 15%가 장애를 가지고 있으며, 이 중 상당수가 상지 기능 제약으로 인한 컴퓨터 접근 어려움을 겪고 있다. 기존의 보조 기술 솔루션들은 높은 비용과 복잡한 설치 과정이라는 근본적인 한계를 가지고 있다. 상용 시선 추적 장치(Eye Tracker)는 수천 달러의 비용을 요구하며, 전용 하드웨어 설치와 전문적인 설정 과정이 필요하다. 이러한 경제적·기술적 진입 장벽은 디지털 격차를 심화시키고 있다.

기존 연구들은 주로 고성능 하드웨어나 복잡한 알고리즘에 의존하는 접근 방식을 취해왔다. 뇌-컴퓨터 인터페이스(Brain-Computer Interface)는 높은 정확도를 제공하지만 침습적 절차와 고비용의 장비를 요구한다. 기존의 얼굴 추적 시스템들은 전용 카메라나 GPU 연산을 필요로 하여 일반 사용자의 접근성을 제한했다. 특히 상용 솔루션들은 폐쇄적 생태계를 구축하여 사용자의 개별적 요구사항에 대한 맞춤화를 어렵게 만들었다. Microsoft의 Windows 10 Eye Control나 Tobii Eye Tracker와 같은 시스템은 전용 하드웨어를 필요로 하며, 이는 높은 구매 비용과 함께 지속적인 유지보수 비용을 발생시킨다. 이러한 제약들은 보조 기술의 대중적 보급을 저해하는 주요 요인이 되고 있다.

본 연구는 상지 장애인의 디지털 접근성을 개선하기 위해 경제적·기술적 진입 장벽을 제거하는 대체 입력 시스템을 제안한다. 연구의 핵심 목적은 표준 웹캠과 저사양 CPU만을 사용하여 정확하고 실용적인 마우스 대체 솔루션을 개발하는 것이다. 본 연구의 주요 기여는 다음과 같다.

- 첫째, 혁신적인 마우스 대체 입력 기법의 제안이다. Face Mesh 모델과 solvePnP 알고리즘을 결합하여 머리 움직임을 통한 정밀한 커서 제어를 구현하고, Eye Aspect Ratio(EAR) 기반의 눈 깜박임 감지를 통해 클릭 명령을 처리하는 직관적인 상호작용 방식을 개발했다.
- 둘째, 저비용·저사양 환경에서 사용 가능한 실용적 시스템 개발이다. 전용 하드웨어나 GPU 없이 표준 웹캠과 일반적인 CPU만으로 실시간 처리가 가능하도록 최적화했다. AMD Ryzen 5 3500 환경에서 12.4% CPU 사용률과 1.12초의 처리 시간으로 100% 성공률을 달성하여 일반 사용자의 기존 컴퓨팅 환경에서 즉시 활용할 수 있는 실용성을 확보했다.
- 셋째, 오픈소스 공개를 통한 진입 장벽 해소이다. 소프트웨어를 완전한 오픈소스로 공개하여 구매 비용이 필요하지 않으며, 사용자 커뮤니티 기반의 지속적인 개선과 맞춤화를 가능하게 했다. 이는 상용 솔루션의 독점적 생태계를 벗어나 누구나 접근할 수 있는 포용적 기술 환경을 조성한다.

이러한 기여를 통해 본 연구는 단순히 기술적 혁신을 넘어서 사회적 포용성을 실현하는 디지털 접근성 솔루션을 제공한다.

## 관련 연구 및 서비스

대체 입력 방법을 연구한 몇몇 선행 연구는 다음과 같다.

한 연구는 눈-음성 인터페이스(eye-voice interface)로 PC를 제어하는 소프트웨어를 개발했다. 마우스는 얼굴 인식을 통해 인식된 사용자의 눈으로 제어되며, 클릭은 음성 인식을 통해 구현된다. 다른 연구는 얼굴 랜드마크 감지(facial landmark detection)와 음성 명령을 활용하여 컴퓨터 제어를 가능하게 하는 저비용 HCI 시스템을 제시했다. 이 시스템은 얼굴 특징을 지속적으로 분석하여 움직임을 포인터 동작으로 변환하는 소프트웨어 모듈을 포함하며, 클릭과 같은 작업을 위한 음성 명령도 허용한다. 이처럼 연구자들은 인터페이스를 제어하기 위해 신체 움직임, 특히 머리와 눈을 사용하는 방법을 연구해왔다. 더 진보된 접근법으로는 신경 활동을 해석하는 뇌-컴퓨터 인터페이스나 물리적 상호작용을 보조하도록 설계된 로봇 시스템이 있다.

선행 연구와 유사하게, 과거에 유사한 효과를 추구했던 여러 서비스가 있다.

- Smyle Mouse는 일반 웹캠을 사용하여 PC를 제어할 수 있으며, 미소를 지어 마우스와 클릭을 제어하도록 설계되었다. 소프트웨어를 사용하기 위해 한 달에 약 29$의 비용이 든다.
- Kai는 사용자의 손에 착용하여 손과 손가락 움직임을 추적하여 컴퓨터와 IoT 지원 장치를 제어한다. 그러나 이 접근 방식은 여전히 수동 제스처에 의존하기 때문에 손 사용에 불편함을 느끼는 사용자에게는 부적합하다.
- Leap Motion은 적외선 센서와 그 반사 신호를 사용하여 손 제스처를 감지한다. 모니터 앞에 위치하여 사용자가 손 움직임을 통해 장치를 제어할 수 있도록 한다. 그럼에도 불구하고, Kai와 마찬가지로 손 기반 상호작용을 피하고 싶은 개인에게는 적합하지 않다.
- Microsoft의 Windows 10 Eye Control은 Smyle Mouse와 유사한 기능을 제공하지만 추가 하드웨어가 필요하다. 구체적으로, Tobii Eye Tracker에서만 작동하며 표준 웹캠으로는 기능할 수 없다. 이 시스템은 시선을 기반으로 마우스 커서를 제어하고 상호작용을 위해 사용자 지정 UI를 오버레이한다. 예를 들어, 사용자는 "클릭" 버튼을 응시하고 커서를 올린 다음, 클릭을 유발하기 위해 일정 시간 동안 초점을 유지해야 한다. 기본 웹캠으로 작동하는 우리의 솔루션과 대조적으로, 이 방법은 전용 장비가 필요하다.
- LipStick Mouse는 미세한 입술 제스처를 사용하여 전체 마우스 제어를 가능하게 한다. 정전식 센서는 클릭, 더블 클릭, 드래그, 스크롤을 포함한 다양한 기능을 지원하며, 감도와 매핑을 사용자 정의할 수 있다. 그러나 여전히 부피가 큰 추가 하드웨어를 구매하고 설치해야 하므로 모든 사용자에게 실용적이지 않을 수 있다.

n-ABLER Joystick과 같이 상지 장애 사용자를 대상으로 한 여러 서비스가 있다. 그러나 이러한 옵션들은 지속적으로 외부 하드웨어에 의존하며 상당한 재정적 투자를 수반하므로, 많은 사용자의 접근성을 제한한다.

## 얼굴 인식 알고리즘

본 시스템은 실시간 얼굴 인식을 활용하여 hands-free 컴퓨터 제어를 가능하게 한다. 사용자의 얼굴이 감지되면, 시스템은 지속적으로 머리 각도를 추적하고, 각도가 미리 정의된 임계값을 초과할 때 커서를 해당 방향(상, 하, 좌, 우)으로 이동시킨다. 또한, 프로그램은 눈의 감김 정도를 정량적으로 분석한다. 사용자가 의도적으로 몇 초 동안 눈을 감으면 이 제스처는 클릭 명령으로 해석되어, 물리적 접촉 없이 클릭이 가능하다. 이 메커니즘을 통해 사용자는 머리 움직임과 눈 깜빡임만으로 컴퓨터를 탐색하고 상호작용할 수 있다.

이러한 프로그램을 구현하기 위해서는 얼굴 랜드마크 감지 모델, 이 랜드마크를 기반으로 머리 각도를 계산하는 알고리즘, 그리고 사용자의 눈이 감겼는지를 판단하는 또 다른 알고리즘을 통합해야 한다.

![face landmark](landmark.png)

**dlib-HOG + Linear SVM 모델**. dlib 라이브러리에서 제공하는 HOG + Linear SVM 모델은 HOG(Histogram of Oriented Gradients)과 Linear SVM(Support Vector Machine) 분류기를 결합한 사전 훈련된 얼굴 감지 모델이다. HOG 알고리즘은 이미지 내의 밝기 변화와 방향을 포착하여 기울기 정보를 추출하고 이를 특징 벡터로 나타낸다. 이 벡터들은 선형 SVM에 입력되어, 주어진 영역에서 얼굴의 존재 유무를 결정하기 위한 이진 분류를 수행한다.

**SSD**.  OpenCV에서 제공하는 ResNet backbone을 가진 SSD(Single Shot Multibox Detector) 기반의 사전 훈련된 얼굴 감지 모델이다. SSD 모델의 입력 크기는 300×300×3으로 고정되어 있으므로, 입력 이미지는 모델에 전달되기 전에 크기가 조정되어야 한다. 추론 후 출력 좌표는 원래 이미지 크기에 맞게 다시 조정된다.

**dlib-shape predictor 68_face_landmarks**. dlib에서 제공하는 모델은 눈, 코, 눈썹, 얼굴 윤곽선을 포함한 68개의 얼굴 랜드마크를 감지하고 2차원 좌표로 반환한다. dlib은 랜드마크 추출을 담당하기 때문에 위에서 설명한 HOG + Linear SVM 또는 SSD 모델과 같은 얼굴 영역 감지 모델과 함께 사용된다.

**Mediapipe - Face Mesh**. Mediapipe에서 제공하는 이 모델은 메시 구조로 배열된 468개의 얼굴 랜드마크를 감지하고, 깊이 정보를 포함한 3D 좌표를 반환한다. 다른 모델과 달리, Face Mesh는 자체 얼굴 감지 알고리즘을 통합하고 있으므로 별도의 얼굴 감지 단계가 필요 없다. 또한 매 프레임마다 얼굴을 다시 감지하는 대신 이전 프레임을 사용하여 얼굴을 추적하는 방식을 활용하여 추론 시간을 단축하는 대신 메모리를 많이 점유한다.

## 머리 각도 계산 알고리즘

![direction](direction.png)

본 연구에서는 얼굴 랜드마크의 좌표를 기반으로 머리 각도를 계산한다. 이때, 랜드마크 좌표가 2차원인지 3차원인지에 따라 계산 방법이 달라진다.

좌표가 2차원인 경우, 머리의 각도는 수직 축을 따른 특정 얼굴 랜드마크의 상대적 위치를 기반으로 추정된다. 구체적으로, 각도 θ는 얼굴 내부 지점과 외부 윤곽 지점 사이의 수직 거리를 얼굴 길이로 정규화하여 계산된다.

$θ=arcsin(\frac{y_{inside}-y_{contour}}{length})$

여기서 $y_{inside}$와 $y_{contour}$는 선택된 내부 및 외부 얼굴 랜드마크의 수직 좌표를 나타내고, $length$는 정규화에 사용되는 대략적인 얼굴 높이를 의미한다. 이 방법은 수평적인 머리 회전이 특정 얼굴 지점 간의 수직 변위를 유발하며, 이를 각도로 변환할 수 있다는 가정에 기반한다.

좌표가 3차원인 경우, solvePnP 알고리즘을 사용하여 머리 각도를 추정한다. 이 알고리즘은 2D 이미지 좌표, 해당 3D 객체 좌표, 카메라 행렬, 그리고 왜곡 계수(distortion coefficients)를 사용하여 회전 벡터를 계산한다. 본 연구에서는 왜곡 계수를 0으로 가정하고, 카메라 고유 행렬(intrinsic matrix)은 입력 이미지 해상도에 기반하여 정의된다. solvePnP에서 얻은 회전 벡터 $[a, b, c]$를 사용하여, 회전 각도 $\theta$와 회전 벡터의 방향 $v$는 다음과 같이 계산된다:

$$
\theta = \sqrt{a^2+b^2+c^2}, \quad
\nu = 
\begin{bmatrix}
\frac{a}{\theta} \\
\frac{b}{\theta} \\
\frac{c}{\theta}
\end{bmatrix}
$$

여기서 $\theta$는 회전 각도를, $v$는 단위 회전 축을 나타낸다.

## 눈 깜빡임 감지 알고리즘

![ear](ear.png)

본 연구에서는 눈 영역 랜드마크 6개의 수직 및 수평 좌표를 사용하여 눈 감김의 정도를 정량화한다. 계산된 값이 미리 정의된 임계값 아래로 떨어지면 눈이 감긴 것으로 인식된다. 구체적으로, 추출된 좌표는 EAR(Eye Aspect Ratio) 방정식에 대입되어 눈 감김의 정도를 다음과 같이 정량화한다:

$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$

EAR 계산에서, 여섯 개의 눈 랜드마크 P1-6은 눈 주위의 주요 지점을 나타내는 데 사용된다. 구체적으로, P1과 P4는 수평 눈꼬리를, P{2, 3, 5, 6}은 위아래 눈꺼풀 위치에 해당한다. 이 랜드마크는 EAR을 계산하는 데 필요한 수직 및 수평 거리를 측정할 수 있게 한다. 일반적으로 EAR이 0.2보다 작으면 눈이 감긴 것으로 인식된다. 그러나 사용자 간의 개인차가 있으므로, 본 연구에서는 초기 보정 과정에서 각 사용자가 눈을 감고 뜨게 하여 개인화된 EAR 임계값을 자동으로 결정한다.

## 모델 성능 평가

위에서 소개한 다양한 모델들을 조합하여 여러 개의 얼굴 각도 측정 모델을 생성한 후, 이들을 비교하고 가장 좋은 성능을 보인 모델을 사용하기 위해 성능 측정 실험을 진행했다. 총 다섯 가지 모델을 다음과 같이 조사했다.

- 모델 1: Face Mesh를 통해 랜드마크를 추출하고 solvePnP 알고리즘으로 각도를 출력
- 모델 2: Face Mesh를 통해 랜드마크를 추출하고 2차원 알고리즘으로 각도를 출력
- 모델 3: HOG + Linear SVM으로 감지된 얼굴 위치에서 shape_predictor_68_face_landmarks를 통해 얼굴 랜드마크를 추출하고 2차원 알고리즘으로 각도를 출력
- 모델 4: SSD로 감지된 얼굴 위치에서 shape_predictor_68_face_landmarks를 통해 얼굴 랜드마크를 추출하고 2차원 알고리즘으로 각도를 출력
- 모델 5: HOG + Linear SVM으로 얼굴 감지에 실패하면 SSD로 얼굴을 감지하고, shape_predictor_68_face_landmarks를 통해 얼굴 랜드마크를 추출한 후 2차원 알고리즘으로 각도를 출력

평가 항목은 최대 CPU 사용량(%), 메모리 사용량(MiB), 실행 시간(초), 성공률(%), 최대 각도 측정(°)이다. CPU는 AMD Ryzen 5 3500 Matisse를 사용했으며, 메트릭 측정을 위해 머리를 좌우로 반복적으로 기울이는 테스트 비디오를 사용했다. 테스트 비디오의 총 프레임 수는 92프레임이다. 모든 알고리즘은 Python 3.8로 구현되었다.

다양한 얼굴 인식 및 랜드마크 감지 모델의 성능 지표는 표에 자세히 설명되어 있으며, 이는 모델 1이 시스템 요구사항에 대해 최적의 CPU 효율성과 실행 시간을 제공함을 보여준다.

| 모델 | CPU 사용량 (%) | 메모리 사용량 (MiB) | 실행 시간 (s) | 성공률 (%) | 최대 각도 (°) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **12.4** | 32.93 | **1.12** | **100** | **좌: 37.7°, 우: 33.5°** |
| 2 | 14.1 | 42.42 | 1.18 | **100** | 좌: 25.5°, 우: 28.3° |
| 3 | 19.9 | 19.40 | 13.62 | 84.8 | 좌: 14.7°, 우: 17.4° |
| 4 | 53.1 | **19.14** | 3.86 | **100** | 좌: 16.2°, 우: 12.3° |
| 5 | 36.5 | 31.18 | 12.87 | **100** | 좌: 16.2°, 우: 17.4° |

Face Mesh를 통해 랜드마크를 추출하는 모델 1과 2는 다른 모델들에 비해 메모리를 더 많이 사용한다. 그러나 실행 시간 면에서 우수하며, 모델 3에 비해 12배 빠른 결과를 보였다. 또한 얼굴을 측정하기 위해 최대 각도가 큰 것이 중요하므로, Face Mesh를 통해 랜드마크를 추출하고 solvePnP를 통해 각도를 계산하는 것이 가장 적절하다는 결론에 도달했다.

## 사용자 테스트

![user test](user-test.png)

### 인터뷰

프로그램을 대중에게 테스트하게 함으로써, 이전에 고려하지 못했던 부분들을 발견하고 객관적인 관점에서 평가하고자 했다. 사용자 테스트는 이 연구의 목적과 기능을 이해하는 비장애인 6명을 대상으로 진행되었다. 인터뷰 대상자는 웹 검색, 유튜브 시청 등 총 5개 과제를 수행하고, 8개 질문을 받았다.

### 1차 테스트 결과

프레임 드랍(frame drop) 문제를 발견했으며, 마우스가 의도한 것보다 더 멀리 움직이는 것을 경험했다. 또한, 사이드바에 확대 버튼이 있지만 미세한 조작이 어렵고, 사이드바의 버튼들이 인접해 있어 종료 버튼과 도움말 버튼을 실수로 클릭하는 경우가 있었다. 또한 일시 정지 버튼이 사이드바와 카메라 화면을 완전히 숨기지 않고 마우스 방향 제어와 깜빡임 제어에서 벗어나지 않는 문제를 해결해야 한다. 또한 설정 페이지에서 스크롤 감도를 직접 설정할 수 있으면 좋겠다는 의견과, 변수의 범위를 자연수로 통일하면 시스템 사용이 더 직관적일 것이라는 의견을 공유했다.

### 수정 결과

![fps](frames.png)

**프레임 드랍**. 프레임 드랍은 CPU가 현재의 모든 연산을 수행하지 못할 때 발생하며, 본 연구에서 가장 연산량이 많은 방향 예측과 눈 깜빡임 예측이 다른 작업과 함께 발생할 때 일어난다. 이 문제는 제어에 혼란과 지연을 유발할 수 있으므로 해결해야 한다. 우선, 이러한 문제를 해결하기 전에 머리를 회전시킬 때의 프레임 속도를 측정했다. 그 결과, Pyautogui로 방향을 제어할 때 가장 큰 프레임 드랍이 발생하는 것을 확인했다. 그러나 그래프에서 볼 수 있듯이 드랍이 발생할 때의 프레임 속도는 연속적이지 않다. 따라서 프레임 속도를 제한함으로써 문제를 해결했다. 제한은 저전력 CPU를 고려한 최대 프레임 속도인 60 FPS로 설정했다.

**일시 정지 기능**. 이전에는 버튼을 누르면 사이드바는 그대로 있고 카메라 화면만 사라졌다. 수정 후에는 마우스 커서가 일시 정지 버튼 위에서 멈춰 방향 제어를 중단하고, 기능을 재개하기 위해 눈 깜빡임만 인식하는 기능이 추가되었다. 또한, 일시 정지 버튼을 누르면 웹 콘텐츠가 가려지는 것을 최대한 방지하기 위해 사이드바의 크기가 일시 정지 버튼 크기로 축소된다.

**머리 움직임**. 이전에는 각 방향의 임계값 이상으로 머리를 돌리면 마우스가 무조건 움직였다. 그러나 마우스를 멈추고 싶을 때 머리가 원래 위치로 돌아올 때 마우스가 계속 움직이는 문제가 있었다. 이를 해결하기 위해, 현재 프레임의 각도가 이전 프레임에서 계산된 각도에 비해 원점에 3도 이상 접근하면 마우스가 멈추도록 설정했다. 따라서 머리가 원래 위치로 돌아갈 때 마우스가 덜 움직이게 된다.

### 2차 테스트 결과

수정된 기능이 제대로 작동하는지 확인하기 위해, 비장애인 사용자 7명을 대상으로 2차 테스트를 수행했다. 이번 반복에서는 간소화된 검증을 위해 3가지 과제를 완료했으며, 참가자들에게는 "프레임 드랍 발생" 및 "의도치 않은 마우스 움직임"에 대해 추가적으로 질문했다. 참가자들이 시스템에 익숙해지는 데 몇 분이 걸렸지만, 이 초기 적응 기간 후에는 조작이 더 쉬워졌다고 보고했다. 1차 테스트에서 확인된 '멈출 때 마우스가 미끄러지는 현상'과 '프레임 드랍으로 인한 지연'은 해결되었다. 이후 한 참가자가 더 먼 거리에서 시스템을 테스트해 볼 수 있는지 물었고, 시스템이 약 2미터 거리에서도 좋은 인식 성능을 유지하는 것을 관찰했다. 그러나 많은 참가자들은 여전히 작은 버튼과 상호작용하기에는 마우스 움직임이 너무 민감하다고 언급했다. 이로 인해 웹사이트나 브라우저 내의 작은 버튼과 상호작용하기가 어려워, 종종 여러 번의 시도가 필요했다. 참가자들의 공통적인 제안은 머리 움직임의 정도에 따라 커서 이동 속도를 가변적으로 구현하는 것이었다.

### 장애인 사용자 연구

접근성 개선을 더욱 평가하기 위해, 전통적인 마우스 사용을 방해하는 손 떨림을 겪는 사용자와 3차 테스트를 진행했다. 이 세션 동안 사용자는 아래 방향 커서 움직임에 어려움을 느꼈으며, 사이드바와 같은 특정 인터페이스 요소가 상호작용을 방해한다고 지적했고, 특히 유튜브 전체 화면 토글에 접근하는 문제를 언급했다. 또한, 웹툰 보기와 같은 과제를 완료하는 동안, 스크롤 버튼을 반복적으로 클릭하는 것이 불편하다고 느꼈다. 이러한 사용성 문제는 방향 제어 개선과 시각적 방해를 줄이기 위한 UI 조정의 필요성을 시사한다. 참가자는 또한 마우스 전용 상호작용이 여전히 한계가 있음을 지적하며, 시스템의 기능을 키보드 대안을 포함하도록 확장할 것을 제안했다. 이 피드백은 접근성을 향상시키기 위해 다중 모드 입력 옵션을 제공하는 것의 중요성을 강조한다. 향후 개발에서는 키보드 에뮬레이션 기능과 커서 제어 감도 정제를 탐색해야 한다.

## 전문가 피드백

> *Palette팀의 얼굴 인식 활용 대체 입력 프로젝트가 매우 잘 수행되었습니다. 특히, 기존의 연구나 상품들이 제공하지 못했던 여러 기법들을 머신러닝과 딥러닝 기술을 잘 활용해 저렴한 비용으로 사용 가능하도록 새로운 방식을 잘 제안하였다고 판단됩니다. 시간과 비용의 관점에서 기존의 Pre-train된 머신러닝 모델들을 잘 활용하고, dlib과 OpenCV 패키지들 잘 활용하였습니다. 한걸음 더 나아가, 이후에는 직접 여러 face landmark recognition 모델을 활용해 인식률을 높이고, 이후 피드백을 받은 부분을 조금 더 보완해 나간다면, 상품화와 실제 서비스로 내 놓아도 손색이 없을 정도로 훌륭한 서비스가 될 것으로 예상됩니다. 끝으로, 이런 과정과 진행을 모두 github repo에 공개해 이후에도 지속적인 발전이 가능한 오픈소스로 꾸준하고 지속적인 관심을 받을 것으로 기대됩니다.* - Microsoft 김대우 이사

![전문가 의견: Microsoft 김대우 이사](advise.png)

## 결론

본 연구는 상지 장애인을 위한 디지털 접근성 문제를 해결하기 위해 세 가지 핵심 기여를 통한 솔루션을 제시했다.

1. 마우스 대체 입력 기법의 성공적 구현을 통해 기존 물리적 입력 장치의 근본적 한계를 극복했다. Face Mesh 모델과 solvePnP 알고리즘을 결합한 머리 움직임 기반 커서 제어와 EAR 기반 눈 깜박임 클릭 시스템은 자연스럽고 직관적인 상호작용을 실현했다.
2. 저비용·저사양 시스템 개발을 통해 보조 기술의 경제적 접근성을 향상시켰다. 전용 하드웨어를 요구하는 기존 상용 솔루션과 달리 일반 사용자의 기존 컴퓨팅 환경에서 즉시 활용 가능한 실용성을 제공한다.
3. 오픈소스 공개를 통해 보조 기술의 진입 장벽을 해소했다. 완전한 소스 코드 공개로 구매 비용을 제거하고, 사용자 커뮤니티 기반의 지속적인 개선과 맞춤화를 가능하게 했다. 이는 폐쇄적 상용 솔루션의 독점 구조를 벗어나 누구나 접근할 수 있는 포용적 기술 환경을 조성했다.

본 연구의 가장 중요한 성과는 기술적 혁신을 넘어서는 사회적 포용성 실현이다. 경제적 제약으로 인해 보조 기술에 접근하지 못했던 사용자에게 실질적인 디지털 참여 기회를 제공함으로써 디지털 격차 해소에 기여했다. 사용자 테스트를 통해 확인된 시스템의 실용성은 이론적 연구를 넘어 실제 사회 문제 해결에 기여하는 연구의 가치를 보여준다.

현재 시스템은 마스크 착용 사용자에 대한 정확도 감소, 조명 조건 변화에 따른 EAR 측정 오차, 작은 웹 요소 상호작용 시의 정밀도 한계 등의 기술적 제약을 가지고 있다. 또한 키보드 입력 기능의 부재로 인해 완전한 컴퓨터 제어에는 한계가 있다. 그러나 이러한 한계들은 오픈소스 생태계의 특성상 전 세계 개발자 커뮤니티의 협력을 통해 점진적으로 해결될 수 있는 영역이다. 향후 연구는 마스크 얼굴 인식 알고리즘 통합, 조명 변화에 강건한 EAR 계산 방법 개발, 미세 제어를 위한 적응적 감도 조절 시스템 구현에 초점을 맞춰야 한다.

- Github: [denev6/face-mouse-control](https://github.com/denev6/face-mouse-control)  
- 학술지(v1): [얼굴 인식과 Pyautogui 마우스 제어 기반의 비접촉식 입력 기법](https://koreascience.or.kr/article/JAKO202228049092231.page)

## 참고 문헌

- Hye Young Kim. 2022. Improvement of Web Accessibility through Auto-generated OCR Based Alternative Text. Master's thesis. Hanbat University, Daejeon, Korea.
- Ji Eun Seo. 2012. An Empirical Study of the Quality of Assistive Technology for Improving Web Accessibility. Master's thesis. Soongsil University, Seoul, Korea.
- Jun Ho Park, So Ra Jo, and Seong Bin Lim. 2021. Object Magnification and Voice Command in Gaze Interface for the Upper Limb Disabled. Journal of Korea Multimedia Society 24, 7  (July 2021), 903-912.
- P. Ramos, M. Zapata, K. Valencia, V. Vargas, and C. Ramos-Galarza. 2022. Low-Cost Human Machine Interface for Computer Control with Facial Landmark Detection and Voice Commands.  Sensors 22, 23 (2022), 9279. https://doi.org/10.3390/s22239279
- P. Dhamanskar, A. C. Poojari, H. S. Sarwade, and R. R. D'silva. 2019. Human Computer Interaction using Hand Gestures and Voice. In Proceedings of International Conference on  Advances in Computing, Communication and Control (ICAC3). IEEE, 1-6.
- Kyung Tae Hwang, Jong Min Lee, and In Ho Jung. 2020. Remote Control System using Face and Detecting a Drowsy Driver. JIIBC 20, 6 (December 2020), 115-121.
- Chao Yu Chen and Jia Hao Chen. 2003. A computer interface for the disabled by using real-time face recognition. In Proceedings of the 25th Annual International Conference of the  IEEE Engineering in Medicine and Biology Society. IEEE, 4 vols., 3615-3618.
- Rıdvan Karatay, Burak Demir, Ahmet Alp Ergin, and Erkan Erkan. 2024. A Real-Time Eye Movement-Based Computer Interface for People with Disabilities. Smart Health 34 (2024),  100521. https://doi.org/10.1016/j.smhl.2024.100521
- Hyunwoo Kim, Seokhee Han, and Jaehyuk Cho. 2023. iMouse: Augmentative Communication with Patients Having Neuro-Locomotor Disabilities Using Simplified Morse Code. Electronics 12,  13 (2023), 2782. https://doi.org/10.3390/electronics12132782
- Diego Camargo-Vargas, Mauro Callejas-Cuervo, and Andrea Catherine Alarcón-Aldana. 2023. Brain-computer interface prototype to support upper limb rehabilitation processes in the  human body. International Journal of Information Technology 15 (2023), 3655-3667. https://doi.org/10.1007/s41870-023-01280-5
- B. Premchand, Zhiyang Zhang, Kai Keng Ang, Jianxiong Yu, Ivan Ozuem Tan, Jeremiah Pei Wen Lam, Alvin Xin Yi Choo, Audrey Sidarta, Philip Wei Hao Kwong, and Lawrence Hock Chye  Chung. 2025. A Personalized Multimodal BCI-Soft Robotics System for Rehabilitating Upper Limb Function in Chronic Stroke Patients. Biomimetics 10, 2 (2025), 94. https://doi.org/10.3390/biomimetics10020094
- Toshiyuki Yamamoto and Taku Hamaguchi. 2023. Development of an Application That Implements a Brain-Computer Interface to an Upper-Limb Motor Assistance Robot to Facilitate  Active Exercise in Patients: A Feasibility Study. Applied Sciences 13, 24 (2023), 13295. https://doi.org/10.3390/app132413295
- Pedro Isaac Rodríguez-Azar, José Manuel Mejía-Muñoz, Oliverio Cruz-Mejía, Rafael Torres-Escobar, and Luis Virgilio Rosas López. 2023. Fog Computing for Control of Cyber-Physical Systems in Industry Using BCI. Sensors 24, 1 (2023), 149. https://doi.org/10.3390/s24010149
- Haoyu Ren, Tao Liu, and Jie Wang. 2023. Design and Analysis of an Upper Limb Rehabilitation Robot Based on Multimodal Control. Sensors 23, 21 (2023), 8801. https://doi.org/10.3390/s23218801
- Dominik Andreas, Hendrik Six, Annika Bliek, and Philipp Beckerle. 2022. Design and Implementation of a Personalizable Alternative Mouse and Keyboard Interface for Individuals  with Limited Upper Limb Mobility. Multimodal Technologies and Interaction 6, 11 (2022), 104. https://doi.org/10.3390/mti6110104
- Marta Gandolla, Andrea Antonietti, Valentina Longatelli, and Alessandra Pedrocchi. 2020. The Effectiveness of Wearable Upper Limb Assistive Devices in Degenerative Neuromuscular  Diseases: A Systematic Review and Meta-Analysis. Frontiers in Bioengineering and Biotechnology 7 (2020), 450. https://doi.org/10.3389/fbioe.2019.00450
- Denise Gür, Nikolas Schäfer, Matthias Kupnik, and Philipp Beckerle. 2020. A Human-Computer Interface Replacing Mouse and Keyboard for Individuals with Limited Upper Limb  Mobility. Multimodal Technologies and Interaction 4, 4 (2020), 84. https://doi.org/10.3390/mti4040084
- Damien Brun, Cédric Gouin-Vallerand, and Sébastien George. 2024. Design and Evaluation of a Versatile Text Input Device for Virtual and Immersive Workspaces. International  Journal of Human-Computer Interaction (2024), 1-46. https://doi.org/10.1080/10447318.2024.2355616
- Justine Kaye O. San Pedro, Ardvin Kester S. Ong, Sean Dominic O. Mendoza, Jose Raphael J. Novela, and Ma. Janice J. Gumasing. 2024. Exploring User Usability Perceptions and  Acceptance of Chording-Enabled Keyboards: A Perspective Between Human-Computer Interaction. Acta Psychologica 250 (2024), 104521. https://doi.org/10.1016/j.actpsy.2024.104521
- Zhen Lu and Ping Zhou. 2019. Hands-Free Human-Computer Interface Based on Facial Myoelectric Pattern Recognition. Frontiers in Neurology 10 (2019), 444. https://doi.org/10.3389/fneur.2019.00444
- Davis E. King. 2009. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10 (2009), 1755-1758. Retrieved from https://github.com/davisking/dlib
- Navneet Dalal and Bill Triggs. 2005. Histograms of oriented gradients for human detection. In Proceedings of the 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). IEEE Computer Society, USA, 886-893. https://doi.org/10.1109/CVPR.2005.177
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 770-778. https://doi.org/10.1109/CVPR.2016.90
- Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. 2016. SSD: Single Shot MultiBox Detector. In Computer Vision - ECCV 2016. Springer International Publishing, 21-37. https://doi.org/10.1007/978-3-319-46448-0_2
- OpenCV team. 2015. OpenCV-Python. Retrieved from https://github.com/opencv/opencv-python
- Google AI Edge team. 2019. MediaPipe. Retrieved from https://github.com/google-ai-edge/mediapipe
- Yury Karynnik, Andrey Ablavatski, Ivan Grishchenko, and Matthias Grundmann. 2019. Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. arXiv preprint arXiv:1907.06724 (2019).
- OpenCV Development Team. 2024. Camera Calibration and 3D Reconstruction. In OpenCV 4.11.0 documentation. Retrieved from https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
- Tereza Soukupova and Jan Cech. 2016. Eye blink detection using facial landmarks. In 21st Computer Vision Winter Workshop. Rimske Toplice, Slovenia.
- Al Sweigart. 2014. PyAutoGUI: Cross-platform GUI automation for human beings. Retrieved from https://github.com/asweigart/pyautogui
