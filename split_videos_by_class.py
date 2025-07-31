import os
import subprocess

# 설정
INPUT_ROOT = "Social_VR"
OUTPUT_ROOT = "Social_VR_Segmented"
SEGMENT_LENGTH = 5  # 초 단위로 자를 길이

# 클래스 디렉토리 순회
for class_name in os.listdir(INPUT_ROOT):
    class_dir = os.path.join(INPUT_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for filename in os.listdir(class_dir):
        if not filename.endswith(".mp4"):
            continue

        input_path = os.path.join(class_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_pattern = os.path.join(output_class_dir, f"{base_name}_clip_%03d.mp4")

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

        print(f"[PROCESSING] {filename} in {class_name}...")
        subprocess.run(cmd)

