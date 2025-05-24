---
title: Support Vector Machine
tags: [Python, AI]
category: Machine-Learning
toc: true 
math: true
media_subpath: /assets/posts/svm/
---

## Linear SVM

`Support Vector Machine`은 분류 문제를 해결하는 머신러닝 기법이다. 대체적으로 준수한 성능을 보이며 `SVM` 또는 `SVC(Support Vector Classifier)`라고 부른다.

### 아이디어

![](svm-how.png)

직선으로 두 종류의 클래스를 분류하는 문제는 어렵지 않다. 그런데 과연 "어떤 경계가 가장 잘 분류했다고 할 수 있을까?", "새로운 데이터가 추가되었을 때, 어떤 경계가 가장 잘 분류해낼까?" 이 질문의 답으로 `Margin`을 제안한다. 

먼저 `Support Vector`를 알아보자. 

![](support-vectors.png)

경계(초평면)와 가장 인접한 양쪽의 데이터들을 `Support Vector`라고 한다. 그리고 경계(초평면)와 수직인 데이터 간 거리를 `Margin`이라고 한다. 

![](margin.png)

이 때 `Margin`이 최대가 되어야 한다. 만약 새로운 데이터가 비슷한 양상으로 들어온다면 어떨까? (위 이미지)오른쪽 분류기는 극단적으로 값을 나눈다. 반면, 왼쪽 경계는 상대적으로 완만하게 범위를 잘 나눈다. 따라서, Margin을 최대화했을 때 경계가 합리적이라고 할 수 있다. 

### 문제 정의

> Margin이 최대가 되는 경계를 찾아라

`Margin`을 정의하고 `Margin`을 최대화하는 문제를 풀면 된다.

![](svm-math.png)

분류를 수행하는 경계를 $wx+b=0$이라고 하자.

양성 쪽 초평면은 $wx+b=1$이며, 그 위에 있는 `support vector`를 $x^{+}$라고 하자.

음성 초평면은 $wx+b=-1$, 그 위의 `support vector`를 $x^{-}$라고 하겠다.

`Margin`은 두 `support vector` 사이의 거리이므로 $ \parallel x^{+}-x^{-}\parallel$로 정의할 수 있다.

그리고 각 `support vector`를 식에 대입하면,

$wx^{+}+b=1 \\ wx^{-}+b=-1$

두 식을 빼면,

$w(x^{+}-x^{-})=2 \\ x^{+}-x^{-}=\cfrac{2}{w}$

따라서, `Margin`을 w에 대해 정의할 수 있다.

$Margin=\parallel x^{+}-x^{-}\parallel =\parallel \cfrac{2}{w}\parallel$

이제 풀어야할 문제는 $max\cfrac{2}{\parallel w\parallel }$이다. 

> 아래 풀이는 안 봐도 관계없다. 조금 더 수학적으로 접근한 내용일 뿐이다. 선형(Linear) 분류에서 중요한 개념은 이게 전부다. 
{: .prompt-info }

### 풀이 (참고)

$max\cfrac{2}{\parallel w\parallel}$를 풀기 위해 필요한 조건이 있다. 

$y_{i}\in\{-1, +1\}$를 클래스라고 하자. (ie. 동그라미/ 네모)

$y_{i}=1$일 때, $wx_{i}+b\geq1$이고

$y_{i}=-1$일 때, $wx_{i}+b\leq-1$이다.

따라서, $y_{i}(wx_{i}+b)\geq1$이 된다.

이제 문제를 다시 정리해보자.

$max\cfrac{2}{\parallel w\parallel} \\ y_{i}(wx_{i}+b)\geq1$

다시 표현하면,

$min\cfrac{1}{2}\parallel w\parallel \\ y_{i}(wx_{i}+b)\geq1$

식을 뒤집게 되면서 최대화(max) 문제가 `최소화(min)` 문제로 바뀐다. 한 단계 더 나가면,

$min\cfrac{1}{2}w^{2} \\ y_{i}(wx_{i}+b)\geq1$

절댓값은 부호를 없애기 위해 사용했으므로 제곱을 해도 동일한 식이 성립한다. 이렇게 돌아온 이유는 `Quadratic programming` 또는 `Lagrange multiplier`를 사용하기 위해서이다. 이 글에서 다루기에는 너무 길어진다. SVM의 핵심은 아니니 생략하겠다.

---

## 오차 허용 (C)

실제 데이터는 `이상치(outlier)`가 존재한다. 

![](errors.png)

정상적인 데이터라면 파란 세모는 음성 쪽 초평면 아래에 위치해야 한다. 하지만 분류에 어긋나는 데이터들이 존재할 수 있다. 이 때 데이터와 초평면 사이의 거리를 $\varepsilon$이라고 하자. (분류하는 경계와의 거리가 아니다.)

SVM 식은 아래와 같이 재정의된다. 

$$
min(\cfrac{1}{2}w^{2}+C\sum_{k=1}^{R}\varepsilon_{k})
$$

`C`는 오차를 얼마나 반영할지를 나타낸다. 이 값은 우리가 정해줘야 한다.

- `C = 0`: 오차를 무시한다. 오차를 최대로 허용한다.
- `C = ∞`: 오차를 최대로 반영한다. 오차를 허용하지 않는다.

`C` 값이 클수록 오차에 민감해지고, 작을수록 오차에 관대해진다.

---

## Non-Linear (Kernel)

![](1d.png)

때로는 선형 경계로는 분류할 수 없는 데이터도 있다. 이럴 때는 데이터 차원을 높여 문제를 해결한다. 

![](2d.png)

위 예시의 경우, 1차원 데이터를 2차원으로 맵핑했다.

$$
\Phi(x): x \to (x, x^{2})
$$

이처럼 차원을 변환하면 해결할 수 있다. 하지만 모든 데이터를 특정 차원으로 맵핑하려면 많은 비용이 든다. 따라서 연산을 줄이는 꼼수를 사용한다.

### Kernel Trick

[Linear-SVM/풀이](#풀이-참고)에서 생략했지만, `SVM`을 푸는 과정에서 두 벡터 간의 점곱을 계산하게 된다. 이때 `Kernel Trick`을 사용하면 연산을 줄일 수 있다.

예를 들어, $x$를 변환하는 $\Phi(x)$를 아래와 같이 정의해보자. 

$$
\Phi(x_{1}, x_{2}):(x_{1}, x_{2})\to(1,\sqrt{3}x_{1},\sqrt{3}x_{2},\sqrt{3}x_{1}^{2},\sqrt{3}x_{2}^{2},x_{1}^{3},x_{2}^{3},\sqrt{6}x_{1}x_{2},\sqrt{3}x_{1}x_{2}^{2},\sqrt{3}x_{1}^{2}x_{2})
$$

그리고 $\Phi(x_{1})\cdot\Phi(x_{2})$를 풀어보자.

$$
\Phi(x_{1})\cdot\Phi(x_{2})=(x_{1}\cdot x_{2}+1)^{3}
$$

결과는 간단하게 정리된다. 따라서 실제로 데이터의 차원을 변환하는 것이 아니라 `점곱`의 계산 결과를 이용해 빠르게 처리할 수 있다. 위 커널 함수를 일반화하면 아래와 같다.

$$
K(x_{1},x_{2})=\Phi(x_{1})\cdot\Phi(x_{2})=(x_{1}\cdot x_{2}+1)^{k}
$$

여러 커널 함수가 있지만 가장 많이 사용되는 건 `RBF(Radial Basis Function)`이다. 

$$
K(x_{1},x_{2})=exp(-\cfrac{\parallel x_{1}-x_{2}\parallel^2}{2\sigma^{2}})
$$

$$
=exp(-\cfrac{\parallel x_{1}\parallel^2}{2})exp(-\cfrac{\parallel x_{2}\parallel^2}{2})\sum_{n=0}^{\infty}\cfrac{(x_{1}\cdot x_{2})^n}{n!}
$$

식을 풀어보면 데이터를 무한차원으로 맵핑하는 모습을 볼 수 있다.

---

## Python 코드

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# SVC 설정
model = SVC(C=1, kernel="rbf")
# SVC 학습
model.fit(X_train, y_train)
# SVC로 예측
y_pred = model.predict(X_test)
# 정확도 출력
accuracy_score(y_test, y_pred)
```

`sklearn`에서 `SVC`를 제공한다. `SVC`의 파라미터를 보면 `kernel`과 `C` 값을 받는다. 자세한 내용은 [공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)를 참고하자.

`kenrel`은 비선형 문제를 해결할 때 사용했던 커널 함수다. 앞에서 소개했던 `RBF`를 적용한 모습이다. 이외에도 `linear`, `poly`, `sigmoid`, `precomputed`가 있다. 

`C`도 오차 허용에서 봤던 그 값이다. 기본값은 1로 설정되어 있다. 

### 실행

`sklearn`의 `iris` 데이터를 이용해 실제 `SVM`을 테스트 해봤다. `iris`는 [붓꽃 데이터](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)로 `0`, `1`, `2` 3종류의 붓꽃 클래스를 가지고 있다.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(
    data=np.c_[iris.data, iris.target],
    columns=["sepal length", "sepal width", "petal length", "petal width", "target"],
)

FEATURES = ["sepal width", "petal length"]

df = df[[*FEATURES, "target"]]
df = df[df["target"] != 1]

X = df[FEATURES]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30
)
```

데이터를 불러오는 코드이다. (잘 몰라도 된다.)

```python
df.head()
```

|index|sepal width|petal length|target|
|:---:|:---:|:---:|:---:|
|0|3\.5|1\.4|0\.0|
|1|3\.0|1\.4|0\.0|
|2|3\.2|1\.3|0\.0|
|3|3\.1|1\.5|0\.0|
|4|3\.6|1\.4|0\.0|

데이터를 처리한 결과를 보자. `sepal width`, `petal length`를 특징으로 가진다. 참고로 데이터 클래스는 `0`과 `2`를 사용했다. 

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
```

```
1.0
```

시각화를 위해 `linear`하게 분류해 보았다. 정확도 100%가 나온다.

### 시각화

결과에 집중하자. (시각화 코드 몰라도 된다.)

```python
import matplotlib.pyplot as plt

t0 = df[df["target"] == 0]
t2 = df[df["target"] == 2]

w = model.coef_[0]
b = model.intercept_
support_vectors = model.support_vectors_

plt.scatter(x=FEATURES[0], y=FEATURES[1], data=t0, c="r", label="0")
plt.scatter(x=FEATURES[0], y=FEATURES[1], data=t2, c="b", label="2")
plt.scatter(
    support_vectors[:, 0], support_vectors[:, 1], c="y", label="support vectors"
)
line = np.linspace(2, 5)
plt.plot(line, -(w[0] * line + b) / w[1], c="b")
plt.legend()
plt.show()
```

![](result.png)

파란점과 빨간점은 원래 데이터다. 노란점은 `support vector`이고, 검정 선은 `SVM`이 분류에 사용한 경계다. 시각화 해봤더니 잘 계산된 모습을 볼 수 있다. 
