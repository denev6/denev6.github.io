---
title: Go를 빠르게 굴리기
tags: [Go]
category: Playground
toc: true
media_subpath: /assets/posts/faster-go/
---

`Go`를 시작한 나는 Go뽕을 느끼기 위해 백준 문제를 Go로 풀어봤다. 그런데 일부 문제는 `Python` 풀이보다 더 느린 결과를 보였다. 뭔가 잘못됐음을 직감했고, 백준에 제출된 고인물들의 코드를 살펴봤다. 그렇게 삽질이 시작됐다. 

**삽질 결과**: 속도 향상

- `공간 확보`: 1.2x
- `빠른 입력`: 17x
- `문자열 합치기`: 71x
- `정규 표현식`: 2.6x

---

## 메모리 공간 확보

```go
s := make([]int, 100)  // 슬라이스, len: 100, cap: 100
s[i] = val  // 값 대입

s := make([]int, 0, 100)  // 슬라이스, len: 0, cap: 100
s = append(s, val)  // 값 대입
```

slice를 생성할 때, 해당 slice에 저장될 값의 범위를 알 수 있다면 미리 메모리에 공간을 확보하는 것이 유리하다. `make`는 slice의 형식을 입력받은 후, len과 cap을 입력받는다. `len`은 슬라이스의 길이로 \[\]int를 3으로 선언하면 \[0 0 0\]이 만들어진다.

중요한 점은 `Capacity`이다. 만약 길이가 3인 슬라이스에 값을 추가해 길이가 4인 슬라이스를 만든다고 하자. 그럼 Go는 새로운 슬라이스를 만들게 된다. 이 때 성능 저하가 발생한다. 불필요한 슬라이스의 재생성을 막기 위해서는 `cap`을 지정하면 된다. `cap`은 미리 메모리 공간을 얼마나 확보해둘지에 대한 값이다. `len`이 3, `cap`이 5라면 \[0 0 0\]을 생성하지만 5개의 값이 들어갈 수 있는 메모리 공간을 확보해둔 상태이다. 따라서 값을 하나 추가해도 새로운 슬라이스를 생성하지 않는다. 

이는 `map`에도 동일하게 적용된다. 다른 점이라면 `map`은 `make`를 할 때 바로 `cap`을 받는다.

```go
make(map[int]bool, 100) // len: 0, cap: 100
```

### 결과

[백준 10815](https://www.acmicpc.net/problem/10815)를 풀이한 결과이다. 문제에서 최대 입력은 500,000개이다.

|cap|풀이 시간|메모리|
|:-:|:-:|:-:|
|500,000|684ms|32576KB|
|100,000|644ms|48360KB|
|0|784ms|50356KB|

다른 문제에서도 `slice`와 `map` 모두 공간을 미리 확보한 풀이가 빠른 모습을 보여줬다. 

### 참고

`배열`을 사용해 정적으로 공간을 확보하는 방법도 있지만 Go에서는 유연하게 길이를 조정할 수 있는 `slice`를 선호한다. 예를 들어, 함수에 값을 넘길 때 배열보다 slice가 더 범용적으로 사용될 수 있다. 

```go
// 모든 int Slice를 받음
func GetSlice(s []int) []int {
	return s
}

// 길이가 3인 int 배열만 받음
func GetArray(a [3]int) [3]int {
	return a
}

func main() {
	// 정상 실행
	a := make([]int, 3)
	a = GetSlice(a)

	// 에러 발생
	s := [5]int{}
	s = GetArray(s) // 배열의 길이가 다름
}
```

---

## 빠른 입력

입력이 많은 문제의 경우, `fmt` 입력 구문은 느리다. `bufio` 패키지를 사용하면 시간을 단축할 수 있다. 

### 결과

[백준 14425번](https://www.acmicpc.net/problem/14425) 문제를 4가지 입력 방식을 이용해 풀이해 보았다. 

| 입력 방식 | 풀이 시간 |
|-|:-:|
| scanner.Scan | 116ms |
| reader.ReadString | 124ms |
| fmt.Fscan(reader, ...) | 268ms |
| fmt.Scan | 시간 초과 (2초 이상) |

어떤 입력을 사용하는가에 따라 시간을 2배 이상 단축하기도 하고, 시간 초과가 발생하기도 한다. 

```go
// 가장 빠른 풀이
var sc = bufio.NewScanner(os.Stdin)

func main() {
	sc.Scan()       
	sentence := sc.Text()
}
```

### 문제

![백준: 틀렸습니다](test-result.png)

[백준 27649](https://www.acmicpc.net/problem/27649)을 풀어보니 분명 `Python`에서 문제가 없었는데 `Go`로 작성하니 계속 `❗틀렸습니다❗`가 나왔다. 그런데 `Scanner`를 `Reader`로 교체하니 문제가 해결되었다. 

```go
const (
	// MaxScanTokenSize is the maximum size used to buffer a token
	// unless the user provides an explicit buffer with Scanner.Buffer.
	// The actual maximum token size may be smaller as the buffer
	// may need to include, for instance, a newline.
	MaxScanTokenSize = 64 * 1024

	startBufSize = 4096 // Size of initial allocation for buffer.
)
```

`scanner.Scan`은 큰 입력을 받지 못한다. `Scanner`를 구현한 [소스코드](https://pkg.go.dev/bufio#Scanner)를 살펴보면 `MaxScanTokenSize`라는 값이 정의되어 있다. 만약 입력의 크기가 64KB보다 크면 문제가 발생할 수 있다. 따라서 값이 클 것으로 예상되면 차선책인 `reader.ReadString`을 사용하는 것이 안전하다.

### 해결책

```go
import (
	"bufio"
	"os"
	"strings"
)

var reader = bufio.NewReader(os.Stdin)

func main() {
	sentence, _ := reader.ReadString('\n')
	sentence = strings.TrimSpace(sentence)
}
```

`NewScanner` 대신 `NewReader`를 사용한다. 참고로 `ReadString`은 마지막 `\n`까지 읽어온다. 따라서 `TrimSpace`를 통해 줄바꿈 문자를 제거해줘야 한다. ~~이거 놓쳐서 많이 틀렸다.~~

---

## 문자열 합치기 (출력)

```go
res := []string{"a", "b", "c"}
fmt.Println(strings.Join(res, "-"))
// a-b-c
```

문자열을 연결할 때 `+` 연산자를 사용할 수도 있지만 느리다. 따라서 `strings.Join`을 사용하면 빠르게 문자열을 이어붙일 수 있다. 첫 인자로 `string Slice`를 입력받고, 두 번째 인자로 문자열 `사이에 삽입할 문자열`을 건네준다. 

`Join` 메서드가 문자열을 합치는 과정을 보면 내부적으로 `Builder`를 사용하고 있다. ([strings.go;line456](https://cs.opensource.google/go/go/+/refs/tags/go1.21.2:src/strings/strings.go;l=456))

```go
words := []string{"a", "b", "c"}

var b strings.Builder

b.WriteString(words[0])
for _, s := range words[1:] {
    b.WriteString("-")
    b.WriteString(words)
}

fmt.Print(b.String())
// a-b-c
```

> A Builder is used to efficiently build a string using Write methods. It minimizes memory copying. The zero value is ready to use. Do not copy a non-zero Builder.
> - [pkg.go.dev](https://pkg.go.dev/strings#Builder)

`Builder`에 대해 알아두면 시간 단축에 많은 도움이 된다. 대표적인 메서드는 아래와 같다.

- `Len`: 축적된 문자열의 길이
- `Reset`: 초기화
- `String`: 축적된 문자를 문자열로 반환
- `WriteRune`: 문자 입력
- `WriteString`: 문자열 입력

### 결과

[백준 1181번](https://www.acmicpc.net/problem/1181) 문제를 풀이한 결과를 보자.

| 문자열 결합 | 풀이 시간 |
|-|:-:|
| builder.WriteString("\\n") | 28ms |
| Println | 884ms |
| += "\\n" | 시간 초과 (2초 이상) |

`WriteString`이 압도적으로 빠른 것을 볼 수 있다. 심지어는 `Python`을 이용한 동일한 풀이도 200ms가 나왔다. 그에 비해 `Go`가 884ms나 시간 초과를 내는 것을 보면 잘못된 문자열 조작이 얼마나 치명적인가를 알 수 있다.

---

## 정규 표현식

`Go`의 정규 표현식인 `regexp`는 비교적 느리다고 알려져 있다. 따라서 직접 `Go` 레벨에서 처리해주는 것이 속도 향상에 도움이 될 수 있다.

```go
// regexp 예시
re := regexp.MustCompile(`[<>\(\)]|&&|\|\|`)
sentence = re.ReplaceAllString(sentence, ` $0 `)
```

### 결과

| 풀이 방식 | 풀이 시간 |
|-|:-:|
| for { switch } | 168ms |
| regexp.ReplaceAllString | 444ms |

[백준 27649](https://www.acmicpc.net/problem/27649)을 풀어본 결과, 복잡한 `regexp`를 사용하는 것보다 `반복문+조건문`으로 직접 구현하는 것이 더 빠른 것을 볼 수 있다. 다만 '백준 2870', '백준 1264'와 같이 간단한 문제는 정규 표현식을 사용해도 성능에 큰 영향은 없었다. 

---

## 함수 인라인

만약 사용 중인 Go의 버전이 1.16 이하라면 함수의 인자와 반환값을 스택에 전달하는 방식을 사용한다. 이로 인해 약간의 성능 저하가 발생할 수 있으므로 간단한 함수라면 인라인 처리하는 것이 유리하다.

> Go 1.17 implements a new way of passing function arguments and results using registers instead of the stack. Benchmarks for a representative set of Go packages and programs show performance improvements of about 5%, and a typical reduction in binary size of about 2%.  
> \- [Go 1.17 Release Notes](https://go.dev/doc/go1.17#compiler)
{: .prompt-info }

그리고 이 부분은 Go 1.17에서 해결되었다.

`백준`과 `leetcode`에서 `Go 1.18`을 사용하고 있기 때문에 큰 문제가 되지 않는다. 굳이 인라인 처리해서 코드를 더럽게 만들지 말자. (23.10.09)

---

## 체크리스트

- [x] 사전에 메모리를 충분히 확보했는가?
- [x] `fmt`로 입력을 받고 있지 않은가?
- [x] `fmt`나 `+`로 문자열을 적고 있지 않은가?
- [x] 복잡한 정규표현식을 사용하고 있지 않은가?
- [x] `Go`가 최신 버전인가?

이 글은 코딩테스트 한정 `Go`를 빠르게 만드는 방법이다.

![](gopher-recover.png)

성능이 크게 중요하지 않다면 `가독성 좋은 코드`, `안정적인 코드`, `수정/확장이 용이한 코드`가 우선이라는 걸 잊지 말자. ~~불쌍한 Gopher를 위해서라도.~~

---

- 이미지 출처: [tottie000/GopherIllustrations](https://github.com/tottie000/GopherIllustrations#-guidelines-for-the-use-of-illustrations)
- The Go gopher was designed by Renée French. Illustrations by tottie.
