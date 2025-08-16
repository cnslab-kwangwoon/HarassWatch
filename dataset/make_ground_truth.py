# make_ground_truth.py
from pathlib import Path
import csv
import sys

ROOT = Path(sys.argv[1])   # 데이터셋 루트로 바꿔주세요
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
OUT_CSV = "ground_truth.csv"

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

rows = []

for room_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "_templates"):
    room_name = room_dir.name
    # 클래스(=하위 폴더) 순회 (예: Benign, Aggressive, ...)
    for cls_dir in sorted(p for p in room_dir.iterdir() if p.is_dir()):
        stage2_label = cls_dir.name
        stage1_label = "Benign" if stage2_label.lower() == "benign" else "Anomaly"

        for v in sorted(cls_dir.rglob("*")):
            if is_video(v):
                # video_path는 ROOT 기준 상대경로로 저장
                rel = v.relative_to(ROOT)
                rows.append({
                    "video_path": str(rel).replace("\\", "/"),
                    "room_name": room_name,
                    "stage1_label": stage1_label,
                    "stage2_label": "" if stage1_label == "Benign" else stage2_label
                })

# CSV 저장
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["video_path","room_name","stage1_label","stage2_label"])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT_CSV}")


