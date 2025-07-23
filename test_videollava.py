import torch
import numpy as np
import av
import sys

from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "LanguageBind/Video-LLaVA-7B-hf"

print("Loading model...")
processor = VideoLlavaProcessor.from_pretrained(model_id)
model = VideoLlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)
print("Model loaded.")

# 비디오 프레임 추출 함수
def extract_frames(video_path, num_frames=8):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > indices[-1]:
            break
    return np.stack(frames)

# CLI 입력 받기
if len(sys.argv) != 3:
    print("Usage: python test_videollava.py <video.mp4> <question>")
    sys.exit(1)

video_path = sys.argv[1]
question = sys.argv[2]

# 추론
print("Extracting frames...")
frames = extract_frames(video_path)

prompt = f"USER: <video> {question} ASSISTANT:"
inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(device)

print("Generating answer...")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)

answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("\n📢 Answer:", answer)

