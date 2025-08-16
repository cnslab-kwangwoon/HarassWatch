import csv
import sys
from pathlib import Path

# 루트 디렉토리 설정
ROOT_DIR = Path(sys.argv[1])  # 필요시 절대경로로 바꿔도 됨
CSV_OUTPUT = sys.argv[1] + "/ground_truth.csv"

rows = []

# 루트 디렉토리 순회
for behavior_class_dir in sorted(ROOT_DIR.iterdir()):
    if not behavior_class_dir.is_dir():
        continue

    # 상위 폴더에서 main label 추출 (예: Aggressive_Behavior → Aggressive)
    main_label = behavior_class_dir.name.split("_")[0]

    # 하위 행동 폴더 순회
    for sub_behavior_dir in sorted(behavior_class_dir.iterdir()):
        if not sub_behavior_dir.is_dir():
            continue

        sub_label = sub_behavior_dir.name

        # .mp4 파일 순회
        for video_file in sorted(sub_behavior_dir.glob("*.mp4")):
            rows.append({
                "video_path": str(video_file),       # 상대경로 또는 절대경로 가능
                "label": main_label,
                "sub_label": sub_label
            })

# CSV 저장
with open(CSV_OUTPUT, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["video_path", "label", "sub_label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV saved to '{CSV_OUTPUT}' with {len(rows)} rows.")
