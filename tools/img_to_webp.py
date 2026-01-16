import os
import argparse
from PIL import Image


def convert_to_webp(target_dir: str, quality: int):
    extensions = (".png", ".jpg", ".jpeg")

    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                img_path = os.path.join(root, filename)
                output_path = os.path.splitext(img_path)[0] + ".webp"
                if os.path.exists(output_path):
                    do_overwrite = input(f"Do u want to overwrite '{output_path}'? (y/n): ")
                    do_overwrite = do_overwrite.strip().lower() == 'y'
                    if not do_overwrite:
                        continue
                try:
                    with Image.open(img_path) as img:
                        img.save(output_path, "webp", quality=quality)
                    os.remove(img_path)
                    print(f"Converted: {img_path} -> {output_path}")
                except Exception as e:
                    print(f"Failed to convert {img_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to WebP format")
    parser.add_argument(
        "dir", nargs="?", default="assets/posts", help="Target directory"
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=90, help="Compression quality (1-100)"
    )

    args = parser.parse_args()
    convert_to_webp(args.dir, int(args.quality))
