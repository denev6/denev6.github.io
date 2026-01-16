# Jin's NOTE

- [denev6.github.io](https://denev6.github.io/)
- 호기심 가득한 개발자 노트

## Posts

```yml
---
title: TITLE
tags: [TAG_1, TAG_2]
category: CATEGORY
toc: true
math: true
pin: true
img_path: /assets/posts/DIR/
---
```

tags: `NLP`, `Vision`, `Multimodal`, `RL`, `HCI`, `Agent`, `Application`, `Python`, `Golang`, `Journey`

## Tools

### Build

```sh
sh tools/build.sh
```

### Serve

```sh
sh tools/serve.sh
```

### Utils

```sh
uv run style_guide/main.py
uv run tools/img_to_webp.py assets/posts
uv run tools/md_to_webp.py _posts
```
