FROM debian:11 
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    python2.7 \
    python2.7-dev \ 
    wget \ 
    build-essential \
    libsndfile1 \
    ffmpeg
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python2.7 get-pip.py && \
    rm get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
RUN pip install 'setuptools<48'
RUN pip install librosa==0.7.2 resampy==0.2.2 llvmlite==0.31.0
RUN groupadd -g 1000 vscode \
  && useradd -u 1000 -g vscode -s /bin/bash -m vscode
COPY . .

