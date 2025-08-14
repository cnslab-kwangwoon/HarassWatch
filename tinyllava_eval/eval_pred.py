import pandas as pd
import re
from sklearn.metrics import classification_report, confusion_matrix
import sys

# 파일 경로 설정
PRED_PATH = sys.argv[1]
GT_PATH = sys.argv[2]

# 결과 로딩
pred_df = pd.read_csv(PRED_PATH)
gt_df = pd.read_csv(GT_PATH)

# 🔍 예측에서 Label 파싱
def extract_label(text):
    if isinstance(text, str):
        # 다양한 형식 커버
        m = re.search(r"(Label:)?\s*(Aggressive|Benign|Personal|Disruptive)", text, re.IGNORECASE)
        if m:
            label = m.group(2).capitalize()
            if label == "Personal":
                return "Personal"
            elif label == "Aggressive":
                return "Aggressive"
            elif label == "Benign":
                return "Benign"
            elif label == "Disruptive":
                return "Disruptive"
    return "Unknown"

# 예측 정리
pred_df["pred_label"] = pred_df["response_text"].apply(extract_label)

pred_df = pred_df.drop(columns=["true_label"])

print(pred_df)

# GT 정리
gt_df = gt_df.rename(columns={"label": "true_label"})
gt_df = gt_df.drop(columns=["sub_label"])

print(gt_df)

# 🔗 merge on video_path
df = pd.merge(pred_df, gt_df, on="video_path")

print(df)

# ❗ Unknown 제외
df = df[df["pred_label"] != "Unknown"]

# 🎯 평가
print("📊 Classification Report:")
print(classification_report(df["true_label"], df["pred_label"], digits=3))

print("\n🧩 Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(df["true_label"], df["pred_label"]),
                   index=["True_" + lbl for lbl in df["true_label"].unique()],
                   columns=["Pred_" + lbl for lbl in df["pred_label"].unique()]))

# 🔢 저장 (선택)
df.to_csv("merged_with_predictions.csv", index=False)

# 📌 오답 사례 추출 및 저장
wrong_cases = df[df["true_label"] != df["pred_label"]]
wrong_cases = wrong_cases.drop(columns=["response_text"])

if not wrong_cases.empty:
    wrong_cases_path = "misclassified_cases.csv"
    wrong_cases.to_csv(wrong_cases_path, index=False)
    print(f"❌ Misclassified examples saved to: {wrong_cases_path}")
    print(wrong_cases[["video_path", "true_label", "pred_label"]].head(10))  # 상위 10개만 미리보기
else:
    print("✅ All predictions are correct!")
