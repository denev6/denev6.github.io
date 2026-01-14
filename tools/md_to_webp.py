import os
import re
import argparse
from pathlib import Path

def update_image_extensions(root_dir):
    root_path = Path(root_dir)

    md_pattern = re.compile(r'(!\[.*?\]\()(.*?\.)(?:png|jpg|jpeg)(\))', re.IGNORECASE)
    html_pattern = re.compile(r'(<img[^>]*?src=["\'])(.*?\.)(?:png|jpg|jpeg)(["\'][^>]*?>)', re.IGNORECASE)

    for md_file in root_path.rglob('*.md'):
        try:
            content = md_file.read_text(encoding='utf-8')

            new_content = md_pattern.sub(r'\1\2webp\3', content)
            new_content = html_pattern.sub(r'\1\2webp\3', new_content)

            if content != new_content:
                md_file.write_text(new_content, encoding='utf-8')
                print(f"수정 완료: {md_file}")
        except Exception as e:
            print(f"파일 처리 오류 ({md_file}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to WebP format")
    parser.add_argument(
        "dir", nargs="?", default="assets/posts", help="Target directory"
    )
    args = parser.parse_args()
    if os.path.isdir(args.dir):
        update_image_extensions(args.dir)
        print("모든 처리가 완료되었습니다.")
    else:
        print("유효하지 않은 경로입니다.")
