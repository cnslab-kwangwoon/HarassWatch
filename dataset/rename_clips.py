import os
import sys

ROOT_DIR = sys.argv[1]  # 🔁 여기를 본인의 루트 경로로 수정하세요
EXTENSIONS = [".mp4", ".avi", ".mov"]

def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in EXTENSIONS)

def rename_all_clips(root_dir):
    count = 1
    for dirpath, _, filenames in os.walk(root_dir):
        filenames = sorted(f for f in filenames if is_video_file(f) and not f.startswith("._"))
        for fname in filenames:
            old_path = os.path.join(dirpath, fname)
            new_name = f"{count:05d}.mp4"
            new_path = os.path.join(dirpath, new_name)
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {old_path} → {new_path}")
            count += 1

if __name__ == "__main__":
    rename_all_clips(ROOT_DIR)
