###FROM python:3.10-slim
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

EXPOSE 8000

CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0", "--port", "8000"]