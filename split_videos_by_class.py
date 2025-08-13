import os
import subprocess
import sys

# 사용법: python split_recursive.py <INPUT_ROOT> <OUTPUT_ROOT> <SEGMENT_LENGTH_SEC>
INPUT_ROOT = sys.argv[1]
OUTPUT_ROOT = sys.argv[2]
SEGMENT_LENGTH = int(sys.argv[3])  # 초 단위

VIDEO_EXTS = (".mp4",)  # 필요하면 (".mp4", ".mov", ".mkv", ".avi") 로 확장

for root, dirs, files in os.walk(INPUT_ROOT):
    # INPUT_ROOT 기준 상대 경로 계산 → OUTPUT_ROOT에 동일 구조로 생성
    rel = os.path.relpath(root, INPUT_ROOT)
    out_dir = os.path.join(OUTPUT_ROOT, rel) if rel != "." else OUTPUT_ROOT
    os.makedirs(out_dir, exist_ok=True)

    for filename in files:
        if not filename.lower().endswith(VIDEO_EXTS):
            continue

        input_path = os.path.join(root, filename)
        base_name, _ = os.path.splitext(filename)
        output_pattern = os.path.join(out_dir, f"{base_name}_clip_%03d.mp4")

        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-i", input_path,
            "-c", "copy",              # GOP 경계에서만 자름(빠름). 정확한 컷이 필요하면 "-c:v", "libx264", "-c:a", "aac" 사용
            "-map", "0",
            "-segment_time", str(SEGMENT_LENGTH),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]

        print(f"[PROCESSING] {input_path} → {output_pattern}")
        subprocess.run(cmd, check=False)
