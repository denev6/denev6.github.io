import re
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

tone_system_prompt_path = Path("style_guide") / "tone_sys.md"
tone_system_prompt = tone_system_prompt_path.read_text(encoding="utf-8").strip()

tone_user_prompt_path = Path("style_guide") / "tone_user.md"
tone_user_prompt = tone_user_prompt_path.read_text(encoding="utf-8").strip()

eval_sys_prompt_path = Path("style_guide") / "eval_sys.md"
eval_sys_prompt = eval_sys_prompt_path.read_text(encoding="utf-8").strip()

eval_user_prompt_path = Path("style_guide") / "eval_user.md"
eval_user_prompt = eval_user_prompt_path.read_text(encoding="utf-8").strip()


def read_document(file_name: str) -> str:
    input_path = Path(file_name)
    content = input_path.read_text(encoding="utf-8")
    return f"""\
<Document>
```markdown
{content}
```
</Document>
"""


def get_token_cost(in_token: int, out_token: int, model: str) -> float:
    model = model.strip().lower()
    if model == "gpt-4o":
        return (in_token * 2.5 + out_token * 10) / 1000000
    if model == "gpt-4o-mini":
        return (in_token * 0.15 + out_token * 0.6) / 1000000
    return -1


def remove_time_stamp(file_name: str) -> str:
    return file_name.split("-", 3)[-1]


def strip_markdown_tags(text: str) -> str:
    # ^```(?:markdown)?\s* : 시작 부분의 백틱과 선택적인 markdown 키워드 제거
    # (.*?) : 본문 내용 추출 (re.DOTALL로 줄바꿈 포함)
    # \s*```$ : 끝 부분의 백틱 제거
    pattern = r"^```(?:markdown)?\s*(.*?)\s*```$"
    match = re.search(pattern, text.strip(), re.DOTALL)

    if match:
        return match.group(1)
    return text.strip()


def write_response(file_name: str, response: str) -> str:
    output_path = Path("style_guide") / "output" / Path(file_name).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = strip_markdown_tags(response)
    output_path.write_text(response, encoding="utf-8")

    return str(output_path)


def rewrite_tone(file_name: str, model="gpt-4o"):
    content = read_document(file_name)

    llm = ChatOpenAI(model=model, temperature=0.7, top_p=0.95, frequency_penalty=0.1)
    response = llm.invoke(
        [
            SystemMessage(content=tone_system_prompt),
            HumanMessage(content=f"{tone_user_prompt}\n{content}"),
        ]
    )

    output_path = write_response(file_name, str(response.content))

    usage = response.usage_metadata
    token_cost = get_token_cost(
        usage["input_tokens"], usage["output_tokens"], model=model
    )
    return output_path, token_cost


def evaluate_sentence(file_name: str, model="gpt-4o"):
    content = read_document(file_name)

    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        top_p=0.85,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    response = llm.invoke(
        [
            SystemMessage(content=eval_sys_prompt),
            HumanMessage(content=f"\n{eval_user_prompt}\n{content}"),
        ]
    )

    p = Path(file_name)
    output_path = f"{p.stem}_comment{p.suffix}"
    output_path = write_response(f"{output_path}", str(response.content))

    usage = response.usage_metadata
    token_cost = get_token_cost(
        usage["input_tokens"], usage["output_tokens"], model=model
    )
    return output_path, token_cost


def main(files: list[str], model="gpt=4o", do_rewrite=True, do_eval=False):
    print(f"Found {len(files)} file(s)!\n")

    for idx, file in enumerate(files, start=1):
        token_cost = 0
        try:
            if do_rewrite:
              _, rewrite_token_cost = rewrite_tone(file, model=model)
              token_cost += rewrite_token_cost
            if do_eval:
              _, eval_token_cost = evaluate_sentence(file, model=model)
              token_cost += eval_token_cost
            print(f"[{idx}] Saved '{remove_time_stamp(file)}' (${token_cost:.6f})")
            time.sleep(3)

        except Exception as exp:
            print(f"[{idx}] {exp} while '{remove_time_stamp(file)}'\n")
            time.sleep(10)


# 사용 예시
if __name__ == "__main__":
    load_dotenv()

    # 예시: `files = ["_posts/playground/2025-12-24-memoir.md"]`
    files = [
        f"_posts/{f}.md"
        for f in (
            "playground/2026-01-12-dev-roo",
        )
    ]
    main(files, model="gpt-4o", do_rewrite=True, do_eval=True)
