FROM python:3.7.12

LABEL Name="paddlefl"
LABEL Version="1.1.2"
LABEL Author="thinkall"

WORKDIR /paddle

# Install paddle_fl
RUN pip install -U --no-cache pip && pip install --no-cache paddlepaddle==1.8.5 paddle-serving-client==0.4.0 zmq paddle_fl==1.1.2

# Install extra dependencies
RUN apt update && apt install patchelf -y && apt clean

CMD python
