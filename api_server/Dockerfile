# Stage 1: Build Stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build

WORKDIR /usr/src/app

# Copy and install dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /usr/src/app

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/torch/lib

# Target for development
FROM build AS dev

#ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/torch/lib

EXPOSE 8000

# Run app in development mode
CMD ["fastapi", "dev", "api_server/app.py", "--host", "0.0.0.0", "--port", "8000"]

# Target for langgraph server model debugging
FROM build AS langgraph

RUN pip install -U "langgraph-cli[inmem]"

#ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/torch/lib

EXPOSE 2024

# Run the langgraph server in dev mode, change default host to localhost 0.0.0.0
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]

# Runtime Stage, does not get built in dev or langraph mode
FROM python:slim-bullseye AS runtime

WORKDIR /usr/src/app

COPY --from=build /usr/src/app ./api_server

COPY --from=build /opt/conda /opt/conda

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.10/site-packages/torch/lib

EXPOSE 8000

# Run the application (for production mode)
CMD ["uvicorn", "api_server.app:app", "--host", "0.0.0.0", "--port", "8000"]