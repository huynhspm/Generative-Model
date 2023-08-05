ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.02-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements_for_docker.txt .

RUN pip install --no-cache-dir -r requirements_for_docker.txt

# Copy the rest of the files
COPY . .

RUN git config --global --add safe.directory /workspace

# Run bash by default
CMD ["bash"]
