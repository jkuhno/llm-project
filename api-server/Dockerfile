# Stage 1: Build Stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build

WORKDIR /usr/src/app

# Copy and install dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /usr/src/app

FROM build AS dev

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/torch/lib

EXPOSE 8000
# Run app in developmetn mode
CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: Runtime Stage
FROM python:slim-bullseye AS runtime

WORKDIR /usr/src/app

COPY --from=build /usr/src/app .

COPY --from=build /opt/conda /opt/conda

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.10/site-packages/torch/lib

EXPOSE 8000

# Run the application (for production mode)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]