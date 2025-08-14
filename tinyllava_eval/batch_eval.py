import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "TinyLLaVA-Video"))

from tqdm import tqdm
import argparse
import re
import requests
import time
import gc
import traceback, sys
import pandas as pd
import io
from PIL import Image
from io import BytesIO

import torch
from transformers import PreTrainedModel
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import functional as F
from torchvision.io import read_video

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def video_parser(args):
    out = args.video_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def save_frames(frames, save_dir="/data/vlm/zxj/others/demo"):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray((frame.cpu().numpy().transpose(1, 2, 0)).astype('uint8'))
        img.save(os.path.join(save_dir, f"frame_{i}.png"))

def _ts():
    return time.strftime("%H:%M:%S")

def parse_prediction(output_text):

    # 1. Label: 형식 우선 탐지
    label_match = re.search(r"Label:\s*(Aggressive|Benign|Personal)", text, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).capitalize()
    else:
        # 2. fallback: 자연어 기반 서술 탐색
        label_match_alt = re.search(r"labeled\s+'?(Aggressive|Benign|Personal)'?", text, re.IGNORECASE)
        label = label_match_alt.group(1).capitalize() if label_match_alt else None

    # 3. Reason 추출 (Label: 다음 문장 or 'because ~' 절)
    reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # fallback
        reason_match_alt = re.search(r"because\s+(.+?)[\.\n]", text, re.IGNORECASE)
        reason = reason_match_alt.group(1).strip() if reason_match_alt else None

    return label, reason

def eval_model(args, model=None, tokenizer=None, image_processor=None):
    # Model
    t0_total = time.time()
    # [LOG]
    print(f"[{_ts()}] ▶ eval_model start | video={args.video_file} image={args.image_file} "
          f"| frames target={args.num_frame} max_new_tokens={args.max_new_tokens}", flush=True)

    # 모델을 외부에서 로드해서 전달받았다면 스킵
    if model is None or tokenizer is None or image_processor is None:
        t0_load = time.time()
        if args.model_path is not None:
            model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
        else:
            assert args.model is not None, 'model_path or model must be provided'
            model = args.model
            if hasattr(model.config, "max_sequence_length"):
                context_len = model.config.max_sequence_length
            else:
                context_len = 2048
            tokenizer = model.tokenizer
            image_processor = model.vision_tower._image_processor
        # [LOG]
        print(f"[{_ts()}] ✅ model loaded in {time.time()-t0_load:.2f}s", flush=True)

    qs = args.query
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # [LOG]
    t0_prep = time.time()
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_preprocess = ImagePreprocess(image_processor, data_args)
    video_preprocess = VideoPreprocess(image_processor, data_args)
    #print(f"[{_ts()}] 📄 text/image/video preprocessors ready ({time.time()-t0_prep:.2f}s)", flush=True)

    model.cuda()

    msg = Message()
    msg.add_message(qs)

    t0_tok = time.time()
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    prompt = result['prompt']
    input_ids = input_ids.unsqueeze(0).cuda()
    # [LOG]
    #print(f"[{_ts()}] 🔤 tokens prepared: shape={tuple(input_ids.shape)} in {time.time()-t0_tok:.2f}s", flush=True)

    images_tensor = None
    video_tensor = None
    if args.image_file is not None:
        t0_img = time.time()
        image_files = image_parser(args)
        images = load_images(image_files)[0]
        images_tensor = image_preprocess(images)
        images_tensor = images_tensor.unsqueeze(0).half().cuda()
        # [LOG]
        #print(f"[{_ts()}] 🖼️ image preprocessed: shape={tuple(images_tensor.shape)} "
              #f"({time.time()-t0_img:.2f}s)", flush=True)

    if args.video_file is not None:
        t0_vid = time.time()
        #print(f"[{_ts()}] 🎞️ loading video: {args.video_file}", flush=True)
        video = EncodedVideo.from_path(args.video_file, decoder="decord", decode_audio=False)
        duration = video.duration
        video_data = video.get_clip(start_sec=0.0, end_sec=duration)
        video_data = video_data['video'].permute(1, 0, 2, 3)

        total_frames = video_data.shape[0]
        if args.num_frame > 0:
            frame_indices = np.linspace(0, total_frames - 1, args.num_frame, dtype=int)
        else:
            num_frames_to_extract = min(args.max_frame, max(1, int(duration)))
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
        video_data = video_data[frame_indices]

        # [LOG]
        #print(f"[{_ts()}] 🎯 frames selected: total={total_frames} -> used={len(frame_indices)} "
              #f"({time.time()-t0_vid:.2f}s)", flush=True)

        t0_vprep = time.time()
        videos = []
        for video in video_data:
            video = video_preprocess(video)
            videos.append(video)
        video_tensor = torch.stack(videos)
        #video_tensor = video_tensor.unsqueeze(dim=0)
        video_tensor = video_tensor.unsqueeze(0)                # [1, T, 3, H, W]
        video_tensor = video_tensor.half().contiguous()         # dtype/메모리 정리
        video_tensor = video_tensor.cuda(non_blocking=True)     # GPU로 확실히 이동
        #print(f"[{_ts()}] 🧪 video tensor ready: shape={tuple(video_tensor.shape)} "
              #f"in {time.time()-t0_vprep:.2f}s", flush=True)

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # [LOG]
    t0_gen = time.time()
    '''
    print(f"[{_ts()}] 🚀 generation start...", flush=True)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            video=video_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # [LOG]
    '''

    # generate 직전 동기화 & 로그
    #print(f"[gen] start {time.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
    #torch.cuda.synchronize()

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                video=video_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
    except Exception as e:
        # 예외 세부를 STDERR로 즉시 출력
        print("[gen] EXCEPTION!", file=sys.stderr, flush=True)
        traceback.print_exc()  # 전체 스택
        # CUDA 상태 정리
        try: torch.cuda.synchronize()
        except: pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 상위로 올려서 배치 루프의 재시도/기록 로직이 실행되게
        raise

    torch.cuda.synchronize()
    #print(f"[gen] done  {time.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)

    gen_sec = time.time()-t0_gen
    #print(f"[{_ts()}] ✅ generation done in {gen_sec:.2f}s", flush=True)

    t0_dec = time.time()
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"outputs: {outputs}")
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # [LOG]
    #print(f"[{_ts()}] 📝 decode done in {time.time()-t0_dec:.2f}s", flush=True)

    # [LOG] total
    #print(f"[{_ts()}] ⏱️ total elapsed: {time.time()-t0_total:.2f}s", flush=True)

    return outputs

if __name__ == "__main__":

    MODEL_PATH   = "Zhang199/TinyLLaVA-Video-Qwen2.5-3B-Group-16-512"
    CONV_MODE    = "qwen2_base"
    CSV_PATH     = "ground_truth.csv"
    OUTPUT_PATH  = "tinyllava_batch_results.csv"
    DATASET_PATH = "../dataset/0813Data_20"

    # PROMPT_v*.txt 읽기
    with open("prompts/PROMPT_v3.txt", "r", encoding="utf-8") as f:
        QUERY_TEXT = f.read().strip()

    # Args 공통값
    class Args:
        model_path     = MODEL_PATH
        model          = None
        image_file     = None
        video_file     = None
        query          = QUERY_TEXT
        conv_mode      = CONV_MODE
        sep            = "|"
        temperature    = 0.5
        top_p          = 0.95
        num_beams      = 1
        num_frame      = 16
        max_frame      = 16
        max_new_tokens = 256

    disable_torch_init()

    # --- 모델 한 번만 로드 ---
    model, tokenizer, image_processor, _ = load_pretrained_model(MODEL_PATH)
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True
    #torch.set_float32_matmul_precision("high")

    df = pd.read_csv(DATASET_PATH + "/" + CSV_PATH)
    results = []
    SAVE_EVERY = 20  # 20개마다 저장
    partial_output_path = OUTPUT_PATH.replace(".csv", "_partial.csv")

    #for idx, row in df.iterrows():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
        args = Args()
        args.video_file = DATASET_PATH + "/" + row["video_path"]

        try:
            output_text = eval_model(args, model, tokenizer, image_processor)
            #pred_label, reason = parse_prediction(output_text)
            #print(f"pred_label: {pred_label}, reason: {reason}", flush=True)
            status = "OK"
        except Exception as e:
            output_text = f"[ERROR] {e}"
            pred_label, reason = "Error", f"[ERROR] {e}"
            status = "ERR"

        results.append({
            "video_path": row["video_path"],
            "true_label": row.get("true_label", row.get("label", "")),
            "response_text": output_text
        })

        # 🔁 일정 간격마다 중간 저장
        if (idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(partial_output_path, index=False)
            print(f"📝 Partial save at {idx + 1}: {partial_output_path} ({time.strftime('%X')})", flush=True)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved: {OUTPUT_PATH}")
