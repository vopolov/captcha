FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /captcha
COPY data.py data.py
COPY model.py model.py
COPY inference.py inference.py
COPY weights.pt weights.pt
ENTRYPOINT ["python", "inference.py"]
