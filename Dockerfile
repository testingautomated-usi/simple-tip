FROM tensorflow/tensorflow:2.6.1-gpu

ADD requirements.txt requirements.txt

RUN apt-get -y update && apt-get install -y git

RUN pip install -r ./requirements.txt

# Copy source files
COPY ./src ./src
COPY ./tests ./tests
COPY ./datasets ./datasets
COPY ./reproduction.py ./reproduction.py

