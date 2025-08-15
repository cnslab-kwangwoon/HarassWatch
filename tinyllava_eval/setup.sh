bin/bash
set -e

echo "🟢 Setting up TinyLLaVA-Video eval environment..."

# 1. Conda env 생성
ENV_NAME=tinyllava_video
PYTHON_VERSION=3.10

echo "➡️ Creating conda env: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# 2. 필요한 패키지 설치
echo "➡️ Installing required packages..."
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate peft moviepy av pandas

# 3. TinyLLaVA-Video 레포 clone 및 설치
echo "➡️ Cloning TinyLLaVA-Video..."
git clone https://github.com/ZhangXJ199/TinyLLaVA-Video.git
cd TinyLLaVA-Video
pip install -e .

cd ..
echo "✅ Done! Activate with: conda activate $ENV_NAME"
