FROM python:3

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#Downaload lang models
RUN ["python", "-m", "spacy", "download", "en_core_web_lg"]

#Only copy what we need
COPY data data
COPY main.py main.py

EXPOSE 5000/tcp
CMD [ "python", "-m" , "main"]