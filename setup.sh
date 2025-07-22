#!/bin/bash

# 로그 저장 시작
LOG=/provision.log
echo "[START] Provisioning at $(date)" >> $LOG 2>&1

# Conda 환경 셸 스크립트 로드
source /root/miniconda3/etc/profile.d/conda.sh >> $LOG 2>&1

# 가상환경 생성 (이미 있으면 생략)
conda create -n videollava python=3.10 -y >> $LOG 2>&1
conda activate videollava >> $LOG 2>&1

# 필수 도구 설치
apt update && apt install -y git ffmpeg curl >> $LOG 2>&1

# 코드 클론
git clone https://github.com/PKU-YuanGroup/Video-LLaVA >> $LOG 2>&1
cd Video-LLaVA

# pip 업그레이드
pip install --upgrade pip >> $LOG 2>&1

# 패키지 설치
pip install -e . >> $LOG 2>&1
pip install -e ".[train]" >> $LOG 2>&1
pip install flash-attn --no-build-isolation >> $LOG 2>&1
pip install decord opencv-python \
  git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d >> $LOG 2>&1
pip install gdown >> $LOG 2>&1

# 모델 다운로드
gdown 1bRJuqu-rvlwmEPgafoENHVxG_Spj9QK7 >> $LOG 2>&1

echo "[DONE] Provisioning completed at $(date)" >> $LOG 2>&1

