# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM nvcr.io/nvidia/pytorch:22.07-py3 
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY requirements.txt /tmp/requirements.txt

# RUN apt-get update && apt-get install build-essential -y

RUN python -m pip install -U pip && pip install -r /tmp/requirements.txt --no-cache-dir

RUN pip install gdown

RUN pip install "opencv-python-headless<4.3"

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

ENV DATA_DIR=/app/data

COPY src /app/src

WORKDIR /app/src