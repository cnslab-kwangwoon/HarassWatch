import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 파일 경로
RESULT_CSV_PATH = "social_vr_eval_results.csv"

# 데이터 로드
df = pd.read_csv(RESULT_CSV_PATH)

# 라벨 정제
true_labels = df["true_label"].str.strip()
pred_labels = df["predicted_label"].str.strip()

# 평가할 클래스들 (순서를 맞추기 위해 명시)
class_names = ["Aggressive", "Personal", "Benign"]

# 성능 지표 출력
print("📊 Classification Report:")
print(classification_report(true_labels, pred_labels, labels=class_names, zero_division=0))

# 정확도
acc = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Accuracy: {acc:.4f}")

# 혼동 행렬
print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
cm_df = pd.DataFrame(cm, index=[f"True:{c}" for c in class_names],
                        columns=[f"Pred:{c}" for c in class_names])
print(cm_df)

# 결과 저장 (선택사항)
cm_df.to_csv("confusion_matrix.csv")