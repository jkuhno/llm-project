FROM python:3.10-slim

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["gradio", "app.py", "--demo-name=app"]