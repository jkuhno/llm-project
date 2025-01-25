FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/torch/lib

EXPOSE 8000
# TODO> add prod target to run the backend in production mode
CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0", "--port", "8000"]