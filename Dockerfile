FROM ubuntu:18.04

RUN apt-get -y update && apt-get -y upgrade


RUN apt-get install -y python3-pip
#RUN apt-get install -y python3-setuptools


WORKDIR /app


COPY requirements.txt .

RUN pip3 install -r requirements.txt

#RUN virtualenv venv 
#RUN . venv/bin/activate 
COPY . /app

CMD python3 model.py;python3 server.py 

