import os
import torch
import numpy as np
import av
import pandas as pd
from tqdm import tqdm
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# 설정
DATASET_ROOT = "../dataset/0813Data_20/"
CSV_PATH = DATASET_ROOT + "ground_truth.csv"
OUTPUT_PATH = "social_vr_eval_results_wo_background.csv"
# REASON_PATH = "social_vr_eval_reasons_wo_background.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
with open("prompts/PROMPT_v3.txt", "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read().strip()

# 모델 로드
print("🔧 Loading Video-LLaVA...")
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID)
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
).to(DEVICE)
print("✅ Model loaded.")

# 프레임 추출 함수
def extract_frames(video_path, num_frames=8):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        raise ValueError("No frames in video.")
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > indices[-1]:
            break
    return np.stack(frames)

# 추론
df = pd.read_csv(CSV_PATH)
results = []
reasons = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    video_path = DATASET_ROOT + row["video_path"]
    true_label = row["label"]
    full_path = os.path.join(".", video_path)

    print(f"\n▶️ [{idx + 1}/{len(df)}] Processing: {video_path}")

    try:
        frames = extract_frames(full_path)
        print("✅ Frame extraction successful.")

        prompt = f"USER: <video> {PROMPT_TEMPLATE} ASSISTANT:"
        inputs = processor(text=prompt, videos=frames, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        answer_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"🗨️ Raw response: {answer_raw}")

        # ASSISTANT: 태그 이후만 사용
        if "ASSISTANT:" in answer_raw:
            answer = answer_raw.split("ASSISTANT:")[-1].strip()
        else:
            answer = answer_raw.strip()

        # ✅ 중간 저장
        if (idx + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"💾 [{idx + 1}] Interim results saved to CSV.")

    except Exception as e:
        answer = f"[ERROR] {e}"
        pred = "Error"
        reasoning = f"[ERROR] {e}"
        print(f"❌ Error processing video: {e}")

    results.append({
        "video_path": video_path,
        "true_label": true_label,
        "response_text": answer.strip()
    })

    # reasons.append({
    #     "video_path": video_path,
    #     "reason": reasoning.strip()
    # })

# 결과 저장
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ 저장 완료: {OUTPUT_PATH}")

# pd.DataFrame(reasons).to_csv(REASON_PATH, index=False)
# print(f"\n✅ 저장 완료: {OUTPUT_PATH}, {REASON_PATH}")
