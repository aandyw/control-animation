FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio
RUN python3 -m pip install --no-cache-dir \
        diffusers \
        accelerate \
        transformers

CMD ["/bin/bash"]