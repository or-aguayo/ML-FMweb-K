FROM python:3.9

WORKDIR /service

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "TestML.py"]

#uvicorn main:app --port 10000 --host 0.0.0.0