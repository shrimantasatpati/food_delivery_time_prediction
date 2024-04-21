# syntax=docker/dockerfile:1

# FROM python:3.11.4-slim
ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt

# Download dependencies using cache mount and bind mount
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

EXPOSE 8501

# Copy the source code into the container.
COPY . .

# Run the application.
ENTRYPOINT ["streamlit", "run", "app.py"]
