FROM tensorflow/tensorflow:latest-gpu
COPY ./. /home/.
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install vim -y
RUN apt-get install wget -y

