---
title: CNN 개념과 MNIST 분류
tags: [Python, CV, AI]
category: Study
toc: true 
math: true
media_subpath: /assets/posts/cnn-mnist/
---

기본적인 CNN 모델을 만들기 위해 필요한 개념들을 정리하였다.

결과: [Github: cnn](https://github.com/denev6/deep-learning-codes/blob/main/models/cnn.ipynb)

![](cnn-model.png)
_CNN 모델 구조_

---

## 2D Convolution

`Convolution`은 `합성곱 연산`이다. CNN 모델에서 이미지 특징을 추출하는 과정이 바로 합성곱 연산이다.

-   `Input`: 입력은 (h, w) 크기를 가지는 2차원 이미지.
-   `kernel`: 이미지의 특징을 추출하기 위해 사용되는 필터.
-   `Feature map`: Kernel을 거쳐 연산된 결과로, 추출된 이미지의 특징을 가짐.

2D Convolution 연산은 아래와 같이 수행된다. ("**다음 단계" 클릭**)

<div id="cnn-ex">
    <div class="cnn-data_wrapper">
        <div>
            <div class="cnn-grid" id="input-image">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
            <p>Image (input)</p>
        </div>
        <div>
            <div class="cnn-grid" id="kernel">
                <div>1</div>
                <div>0</div>
                <div>1</div>
                <div>1</div>
            </div>
            <p>Kernel</p>
        </div>
        <div>
            <div class="cnn-grid" id="feature-map">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
            <p>Feature (output)</p>
        </div>
    </div>
    <p id="conv_expr">Image * Kernel = Output</p>
    <div id="conv_wrapper"></div>
    <p id="step">0/4 단계</p>
    <button id="moveKernelBtn">다음 단계</button>
</div>

<style>
    #cnn-ex {
        text-align: center;
        margin: 4em auto;
    }

    .cnn-data_wrapper {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 3rem;
        flex-wrap: wrap;
    }

    .cnn-grid {
        position: relative;
        display: grid;
        gap: 0.25rem;
        padding: 0.25rem;
        color: #000;
    }

    .cnn-grid>div {
        width: 3rem;
        height: 3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px #000 solid;
        border-radius: 0.2rem;
    }

    #input-image {
        grid-template-columns: repeat(3, 3rem);
    }

    #kernel {
        grid-template-columns: repeat(2, 3rem);
    }

    #feature-map {
        grid-template-columns: repeat(2, 3rem);
    }

    #input-image>div {
        background-color: #fff;
    }

    #kernel>div {
        background-color: #e96048;
    }

    #feature-map>div {
        background-color: #fff;
    }

    #conv_expr {
        margin-top: 1em;
        font-weight: 700;
    }

    #conv_wrapper {
        width: 24rem;
        height: 5rem;
        line-height: 5rem;
        margin: 1rem auto;
    }

    #cnn-ex>button {
        background-color: #ddd;
        color: #000;
        margin-top: 1rem;
        padding: 0.25rem 0.5rem;
        border: none;
        border-radius: 0.5rem;
        font-size: 0.8em;
        cursor: pointer;
    }

    #cnn-ex>button:hover {
        background-color: #333;
        color: #fff;
    }

    @media (max-width: 640px) {
        #conv_wrapper {
            width: 18rem;
            padding: 0.25rem 0.5rem;
        }

        .cnn-data_wrapper {
            gap: 1rem;
        }
    }
</style>

`Kernel`은 계속 순회하며 이미지와 합성곱 연산을 수행한다. 그 결과로 추출된 값이 `Feature map`이다. 따라서, Feature map은 이미지로부터 추출된 특징이다. 

위 예시는 (3 x 3) 크기의 이미지와 (2, 2) 크기의 Kernel을 사용했다. Kernel이 우측으로 한 칸씩 그리고 아래로 한 칸씩 움직였다. 이 경우 stride는 (1, 1)이다. `stride`란 kernel이 몇 칸씩 움직이며 합성곱 연산을 수행할 것인지를 뜻한다. stride가 (2, 1)이라면 우측으로 2칸, 아래로 1칸씩 움직인다. 

---

## Conv2d

```python
torch.nn.Conv2d(
    in_channels, 
    out_channels, 
    kernel_size, 
    stride=1, 
)
```

Pytorch에서는 `nn.Conv2d`라는 이름으로 Convolution 객체를 제공한다. `kernel_size`와 `stride`는 위에서 살펴봤던 값이다. 중요한 것은 `입력 채널`과 `출력 채널`을 반드시 입력해 주여야 한다. 

`in_channels`는 입력 이미지 차원을 의미한다. 일반적으로 흑백 이미지는 1, 색상(RGB) 이미지는 3이 된다. 

![](rgb.png)

`out_channels`은 다음 은닉층으로 전달할 출력 크기이다. 

![](channels.png)

위 예시는 흑백 이미지를 입력으로 받으므로 `in_channels`는 1이며, 출력으로 4개의 `Feature map`이 만들어지므로 `out_channels`가 4인 예시다. Conv2d에서는 (out\_channels, in\_channels) 크기의 Kernel을 만들어 out\_channels개의 출력을 만들어낸다. ~~CNN을 만들기 위해 크기가 어떻게 변환되는지 반드시 이해해야 한다.~~

---

## Dilated Convolution

```python
torch.nn.Conv2d(
    in_channels, 
    out_channels, 
    kernel_size, 
    dilation=1
)
```

`dilation`은 Convolution 연산에서 Kernel의 간격을 조정할 때 사용한다. 

![](dilation.png)

dilation 값을 늘리면, 이미지를 탐색할 때 Kernel 값들의 사이 간격이 커진다. 기본값은 1로 설정되어 여백 없는 Kernel 형태로 탐색된다. dilation이 2인 예시를 보면 (3 x 3) Kernel을 이용해 (5 x 5) Kernel이 커버하는 범위를 탐색하고 있다. Dilational Conv는 이미지를 넓은 범위로 탐색해야 하거나 큰 Kernel을 사용할 여유가 안 될 때 연산 효율을 높여준다. 

---

## ReLU

`ReLU`는 활성화 함수 중 하나로, Conv2d를 거쳐 나온 특징을 조정해준다. 

```python
torch.nn.ReLU(inplace=False)
```

![](relu.png)

$$ReLU(x) = max(0, x)$$

`ReLU`는 0 이하의 특징 값을 모두 0으로 만든다. 자세한 내용은 [활성화 함수](/machine-learning/2022/12/18-activation.md)에서 다룬다. 

---

## 2D MaxPooling

`Max Pooling`이란 특정 범위에서 가장 큰 값을 추출해내는 연산이다. 아래에서 **"다음 단계"** 버튼을 눌러 MaxPooling이 수행되는 과정을 볼 수 있다. 

<div id="maxpool-ex">
    <div class="cnn-data_wrapper">
        <div>
            <div class="cnn-grid" id="input-image-pool">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
            <p>Input</p>
        </div>
        <div>
        </div>
        <div>
            <div class="cnn-grid" id="feature-map-pool">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
            <p>MaxPooling</p>
        </div>
    </div>
    <p id="step-pool">0/4 단계</p>
    <button id="maxPoolBtn">다음 단계</button>
</div>

<style>
    #maxpool-ex {
        text-align: center;
        margin: 4em auto;
    }
    #input-image-pool {
        grid-template-columns: repeat(3, 3rem);
    }

    #feature-map-pool {
        grid-template-columns: repeat(2, 3rem);
    }

    #input-image-pool>div {
        background-color: #fff;
    }

    #feature-map-pool>div {
        background-color: #fff;
    }

    #maxpool-ex>button {
        background-color: #ddd;
        color: #000;
        margin-top: 1rem;
        padding: 0.25rem 0.5rem;
        border: none;
        border-radius: 0.5rem;
        font-size: 0.8em;
        cursor: pointer;
    }

    #maxpool-ex>button:hover {
        background-color: #333;
        color: #fff;
    }

    @media (max-width: 640px) {
        .cnn-data_wrapper {
            gap: 1rem;
        }
    }
</style>


이렇게 지역별 특징 값만 추출하여 모델이 과적합되는 현상을 방지한다. 

```python
torch.nn.MaxPool2d(
    kernel_size, 
    stride=None, 
    padding=0, 
    dilation=1
)
```

`kernel_size`, `stride`, `dilation`은 위에서 봤던 값과 동일한 개념이다.

`padding`이란 MaxPooling을 수행하기 전 가장자리에 0 값을 추가하는 과정을 뜻한다. Pytorch의 경우, 0 값을 채우는 `zero-padding`을 수행한다.

![](padding.png)

---

## 분류

분류 모델은 최종 분류 층을 거쳐 레이블을 예측하게 된다. 이 layer는 주로 `Flatten`, `Linear`, `Dropout`, `Softmax`로 구성되어 있다. 

```python
torch.nn.Flatten(start_dim=1, end_dim=-1)
```

`Flatten` Layer는 추출된 특징 값을 1차원의 데이터로 변환해준다.

![](flatten.png)

Batch size를 고려하지 않았을 때, (32, 7, 7)의 크기를 가진 데이터를 (32 x 7 x 7 =)1568의 1차원 데이터로 변환해주는 식이다. 이렇게 변환된 데이터는 `Linear` Layer에 들어가 분류 문제를 해결하는 데 사용된다. 

```python
torch.nn.Linear(in_features, out_features, bias=True)
```

`Linear`는 완전 연결층으로 `Fully Connected Layer`로 불린다. 이전 은닉층에서 들어온 입력값이 이후 은닉층과 모두 연결될 수 있도록 하는 역할을 한다. 마지막에 사용되는 Linear layer는 레이블 개수에 맞춰 값의 크기를 변환해주는 역할을 한다.

> 모델 출력값은 각 레이블일 확률(또는 logit) 값이다. 따라서 레이블이 6개인 분류 문제를 푼다면 출력도 6개여야 한다. 모델 내부에서 어떤 과정을 거쳤든 마지막 값은 레이블 개수가 되도록 조정해야 한다. Linear는 자유롭게 출력 크기를 조정할 수 있다. 따라서 마지막에 Linear를 붙여준다.
{: .prompt-tip }

![](linear.png)

만약 모델의 출력 값이 6개이고 정답 레이블이 2개라면, `Linear(6, 2)`와 같은 형태로 사용되어 최종적으로 2개의 값을 반환하도록 조정한다. 

$$y = W^Tx + b$$

```python
torch.nn.Dropout(p=0.5)
```

`Dropout`은 p(확률 값)에 따라 입력 값 일부가 랜덤하게 0으로 출력된다. 나머지 값들은 Scale factor를 곱한 결과로 출력된다. 이를 통해 일반화 성능을 높이는 효과를 가진다. 대신 train 모드에서만 해당 층이 활성화되고, eval 모드에서는 Dropout이 적용되지 않는다. (입력값과 동일한 출력 값을 가진다.)

$$Dropout(x)=0 \text{ or } \cfrac{1}{1-p}x$$

```python
torch.nn.Softmax(dim=None)
```

`Softmax`는 모델의 출력 값(logits)을 확률 값으로 변환해 준다. Linear를 거쳐 나온 모델의 최종 결과는 \[-∞, +∞\] 범위를 가지는 logit 값이다. 따라서 출력 값을 \[0, 1\]의 범위로 조정해 확률 값을 얻고 싶을 때 softmax를 사용한다. 

$$Softmax(x_i)=\cfrac{exp(x_i)}{\sum_{j}^{C}exp(x_j)}$$

```python
logits.argmax(dim=-1)
```

`argmax`는 가장 큰 값의 인덱스를 반환한다. 최종 출력 값 중 가장 큰 값을 가지는 인덱스를 예측 레이블로 간주한다. 

---

### MNIST

`MNIS`T 데이터셋은 손글씨 숫자 이미지로 이미지 분류 연습에 사용되는 대표적인 데이터다. 아래 코드는 CNN 중심의 코드만 기록하였고, 전체 코드는 [Github: cnn](https://github.com/denev6/deep-learning-codes/blob/main/models/cnn.ipynb)에서 확인할 수 있다. Feature map 등 내부 과정이 자세히 기록되어 있으니 확인하는 걸 추천한다. 

- 전체 코드 보기: [Github: cnn](https://github.com/denev6/deep-learning-codes/blob/main/models/cnn.ipynb)

MNIST 이미지 데이터는 (28 x 28) 크기의 2차원 이미지다. 0 ~ 9까지의 숫자 이미지를 가지고 있으므로 Label의 개수는 10개다. 

![](cnn-model.png)

```python
class CNN(nn.Module):
    def __init__(self, num_label=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(7 * 7 * 32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, num_label)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out
```

위에서 봤던 Conv2d + ReLU + MaxPool2d, Flatten + Linear + Dropout을 모두 적용한 모습이다. 완성된 모델의 구조는 아래와 같다. 학습이 잘 되는지 확인하기 위해 channel과 같은 값들은 임의로 설정하였다. 

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 32, 12, 12]           4,640
              ReLU-5           [-1, 32, 12, 12]               0
         MaxPool2d-6             [-1, 32, 7, 7]               0
            Linear-7                   [-1, 32]          50,208
              ReLU-8                   [-1, 32]               0
           Dropout-9                   [-1, 32]               0
           Linear-10                   [-1, 10]             330
================================================================
Total params: 55,338
Trainable params: 55,338
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.27
Params size (MB): 0.21
Estimated Total Size (MB): 0.49
----------------------------------------------------------------
```

> 위 결과는 torchsummary를 통해 출력한 결과다. torchsummary는 제작한 모델의 구조를 요약해준다. Output-Shape에서 -1로 표현된 부분은 Batch 크기다.
{: .prompt-info }

최종적으로 결과 값을 확인해보면 이미지에 대해 잘 예측하는 것을 볼 수 있다. 

```python
logits = model(images)
probs = F.softmax(logits)
pred = probs.argmax(-1)
```

```
Logits: -7.427, 8.698, 0.407, -4.990, -4.264, -3.783, -4.286, -3.159, -2.024, -4.415
 Probs:  0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
  Pred:  1
  True:  1
```

![](one.png)

---

## Keras

`Keras`로 2D CNN을 선언하는 방식은 아래와 같다.

```python
import tensorflow as tf

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
    )
)
model.add(
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),  
        strides=(1, 1),
    )
)
```

Pytorch와 달리 activation 함수는 `Conv2D` 객체를 선언하며 함께 지정하게 된다.

<script type="text/javascript" src="/scripts/cnn.js"></script>
