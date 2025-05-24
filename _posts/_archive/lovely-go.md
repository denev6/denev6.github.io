---
title: Go 언어의 매력
tags: [Go]
toc: true 
category: Playground
media_subpath: /assets/posts/lovely-go/
---

![](toon1.png)
_@waterglasstoon_

평소 `Python`으로 일상생활에 필요한 프로그램을 만들어 왔다. 하지만 실행 속도가 아쉽다 생각했고, 그 대안으로 `C++`, `Go`를 둘러보던 중 `Go`에 제대로 빠졌다.

![](gopher.png)

우리의 귀여운 `Gopher`다. 마스코트는 어벙해보이지만 성능은 야무지다. 

---

## Why Go?

`Python`을 메인으로 사용하며 아래 조건을 만족하는 서브 언어를 선택해야 했다. 저수준의 언어로 `C++`이나 `Go`를 고민했지만 Go를 선택하게 된 이유는 다음과 같다. 

### 속도

`Pypy3`보다 빠른 실행 속도를 내기 위해 완전한 `컴파일 언어`가 필요했다. `C++`, `Go` 모두 컴파일 언어이고 대부분의 경우 `Python`에 비해 빠르다. 그런데 Go는 실행 속도뿐만 아니라 컴파일 속도도 빠른 편이다. 

> "컴파일 속도가 빨라 인터프리터 언어처럼 쓸 수 있다"

인용 출처가 [나무위키](https://namu.wiki/w/Go(%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D%20%EC%96%B8%EC%96%B4)#s-3.1)라 믿음이 안 가겠지만 써보니 맞는 말이다. 

### 간결한 문법

> Build simple, secure, scalable systems with Go
> \- [*go.dev*](https://go.dev/)

Go는 지원하는 문법이 간결하며 최소한의 키워드만 가진다. 예를 들어 반복문은 `for` 하나만 제공하며, OOP는 가능하지만 `class`를 지원하지 않는다. 지원하는 키워드가 적다고 할 수 있는 게 적다는 뜻은 아니다. 기존 키워드로 해결 가능하다. 

```go
// while 문
for stdin != nil {}

// class
type Object struct {
    id int
}
```

Go는 미니멀한 삶을 추구한다.

### 정적 타입 선언

`Go`는 정적타입 언어다. 최근 `Typescript`에 대한 반발이 생기는 걸 보면 정적 타입이 장점인가에 대해서는 의견이 갈릴 듯하다.

![Js에 불평하는 Ts](toon2.png)
_@waterglasstoon_

하지만 타입이 의도하지 않은 방향으로 처리돼도 "어쩌라고?"를 시전하는 `Python`을 떠올려보자... 동적 타입핑은 데이터가 찢겨져도 아무 일 없다듯 해맑게 돌아간다. 그렇다고 매번 `assert`할 수는 없으니 서브 언어는 정적 타입 언어를 쓰기로 마음 먹었다.

```go
var := 3 // int
```

게다가 Go는 타입을 작성하지 않아도 알아서 추론해준다. 덕분에 중요하지 않은 데이터는 신경쓰지 않고 넘어갈 수 있다. 

### 낮은 진입장벽

일단 `C++`보다는 쉽다. 그리고 저수준 언어치고 코드가 간결하다.

```go
import "fmt"

func main() {
    fmt.Println("hello world")
}
```

### GC 지원

`Garbage Collector`, GC가 자동적으로 메모리를 관리해 준다. GC로 인해 성능 저하가 발생할 수 있다. 그래도 메모리 누수가 발생하는 것보다는 낫다고 생각한다. 이 부분은 `C++`과 차별되는 점이다. 

![C++: 메모리 줏어 병*아](toon3.jpeg)
_@system32comics_

GC 없이 `C++`은 좀 무섭다.

---

## Go의 단점?

위 이유들로 Go를 선택했지만 Go도 분명한 단점이 있다.

-   **활용 범위가 적음**: 많은 국내 대기업 코테에서 `Go`를 허용하지 않는다. 만약 인공지능 개발을 위해서라면 현실적으로 `C++`을 공부하는 게 가성비가 좋긴하다.
-   **부족한 자료**: 특히 국내에서는 아직 `Go`를 많이 사용하지 않다보니 참고할 수 있는 자료가 부족하다. 마이너한 언어다보니 영어 자료도 많은 편은 아니다.

코테의 경우는 `Python`을 주로 사용하기 때문에 문제가 없다. 또 `Go`로 쓴 코드를 `Python` 코드로 옮기는 건 어렵지 않았다. 그럼에도 `C++`에 비해 실용성이 떨어진다는 건 사실이다. 하지만 '_시간이 있을 때 해보고 싶은 걸 더 열정있게 공부하자_'라는 생각이 컸다. 그래서 중간에 `C++`로 넘어가더라도 우선은 `Gopher`(Go 개발자)를 체험해보기로 결정했다. 

---

## Go 후기

`Go`에 익숙해지기 위해 기초적인 `출력문`, `반복문`부터 차근차근 익혀봤다. 

> [github.com/denev6/archive/learn-go](https://github.com/denev6/archive/tree/main/learn-go)

저수준의 언어는 이해 수준에 따라 성능 차이가 많이 난다. Go도 예외는 아니다. 같은 알고리즘을 `Go`로 짜도 `Python`보다 느린 경우가 있었다. 하지만 Go스럽게만 써도 성능 하나만큼은 기가 막힌다. 한 마디로 정리하면, 성능과 가독성을 균형있게 챙긴 언어가 아닌가 싶다. 

Go에 대해 알고 싶다면 [Effiective Go](https://go.dev/doc/effective_go)라는 문서가 있다. Go를 Go스럽게 쓰는 방법이다. 성능을 위해 삽질했던 내용은 ["Go를 빠르게 굴려보자"](/playground/2023/10/08/faster-go.html)에 기록해뒀다. 만약 Go에 대해 찾고 싶다면 `Go`보다 `Golang`으로 검색하면 정확한 결과를 볼 수 있다. 

---

다들 Go하세요.

이미지 출처

- [데브 경수 @waterglasstoon](https://www.instagram.com/waterglasstoon)
- [egonelbre/gophers: CC0-1.0](https://github.com/egonelbre/gophers)
