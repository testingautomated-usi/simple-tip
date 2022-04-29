FROM tensorflow/tensorflow:2.6.1-gpu

RUN \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

ADD requirements.txt requirements.txt

RUN apt-get -y update && apt-get install -y git

RUN pip install -r ./requirements.txt

# Copy source files
COPY ./src ./src
COPY ./tests ./tests
COPY ./datasets ./datasets
COPY ./reproduction.py ./reproduction.py

