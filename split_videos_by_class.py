import os
import subprocess
import sys
import glob

# 사용법: python split_recursive.py <INPUT_ROOT> <OUTPUT_ROOT> <SEGMENT_LENGTH_SEC>
INPUT_ROOT = sys.argv[1]
OUTPUT_ROOT = sys.argv[2]
SEGMENT_LENGTH = int(sys.argv[3])  # 초 단위
MIN_SEGMENT_LENGTH = 20  # 세그먼트 최소 길이(초)

VIDEO_EXTS = (".mp4",)  # 필요시 확장자 추가

def get_video_length(path):
    """ffprobe로 영상 길이(초) 반환"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

for root, dirs, files in os.walk(INPUT_ROOT):
    rel = os.path.relpath(root, INPUT_ROOT)
    out_dir = os.path.join(OUTPUT_ROOT, rel) if rel != "." else OUTPUT_ROOT
    os.makedirs(out_dir, exist_ok=True)

    for filename in files:
        if not filename.lower().endswith(VIDEO_EXTS):
            continue

        input_path = os.path.join(root, filename)
        base_name, _ = os.path.splitext(filename)
        output_pattern = os.path.join(out_dir, f"{base_name}_clip_%03d.mp4")

        # 1️⃣ 세그먼트 자르기
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", str(SEGMENT_LENGTH),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]
        print(f"[PROCESSING] {input_path}")
        subprocess.run(cmd, check=False)

        # 2️⃣ 짧은 세그먼트 삭제
        for seg_path in glob.glob(os.path.join(out_dir, f"{base_name}_clip_*.mp4")):
            seg_length = get_video_length(seg_path)
            if seg_length < MIN_SEGMENT_LENGTH:
                print(f"[DELETE] {seg_path} ({seg_length:.1f}s < {MIN_SEGMENT_LENGTH}s)")
                os.remove(seg_path)
