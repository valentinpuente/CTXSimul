
# Guide to run CTXSimul locally

No source code is provided at this time. Only the basic launcher Python script and the 
swig wrapper to the binary (and the simulator binary). This is only a 
"proof" of life of the paper, not a tool to be used beyond that. Further down the line, the source
code of the simulator will be released.

## Setup a Docker Container 

The binary was built using a Debian11 system. This dockerfile provides a portable 
environment for an x86_64 Linux machine. The python launcher requires librosa 0.7 and python2.7. 
Create a docker container using the following Dockerfile. 

```Dockerfile
FROM debian:11 
WORKDIR /app
RUN apt-get update && apt-get install -y \
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
COPY . .
```

Build the image with:

```bash
sudo docker build -t ctxsimdeb11 .
```

## Access to simulator

Get the binaries of the CTXSimul and boilerplate with:

```bash
git clone [https://github.com/puentev/ctxsimul](https://github.com/valentinpuente/CTXSimul/)
```

Get into the dir and check if can run 

```bash
sudo docker run -it -v "$(pwd)":/app ctxsim:latest /bin/bash
python pyt/speech/realASR.py
```

## Get Audio files (~7GB)
```bash
git clone https://huggingface.co/datasets/vpuente/perezGaldos
```

There is a small set of eleven sentences included in the repo to run the cortex. No analysis scripts are
provided a this time. Take a look inside "./test" to get the stats of the simulation.

```bash
 python pyt/speech/realASR.py -j config1x4.json -c 1000000 -W "10"
 ```

## Run the simulation in batch

Run 8 samples per input size. All simulations are single-thead (requires 40 cores. Uses taskset to 
minimize interference and avoid SMP).

```bash
bash  ./pyt/batchReal.sh -j config1x4.json -W 500 -S "550 1000 2000 5000 10000"  -C 20000000000 -I 0 -R 8
```

To run from a previous cpt, just:

```bash
ln -s <source_model_checkpoint> ./cpt
bash  ./pyt/batchReal.sh  -W 500 -S "550 1000 2000 5000 10000"  -C 20000000000 -I 0 -R 8 -K
```


## Cortex Config

Three different cortices are provided: __config1x4.json__, __config4x4.json__, and __config8x4.json__, which correspond to the
configuration shown in the paper (plus a single row cortex for testing).

In principle, the input data set can be any audio set. Just adjust the configuration inside the Python launcher.

# Using Github Codespaces

Alternatively, the repository comes with a pre-configured devcontainer.
Just start a codespace and it should work from your web browser, using vscode terminal.
