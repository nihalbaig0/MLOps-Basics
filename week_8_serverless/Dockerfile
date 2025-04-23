FROM amazon/aws-lambda-python:latest

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir -p $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error \
    AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    PYTHONPATH="${PYTHONPATH}:./" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Update package manager and install git and gcc-c++
RUN dnf update -y && dnf install -y git gcc-c++

# Copy and install Python dependencies
COPY requirements_inference.txt .
RUN pip install -r requirements_inference.txt --no-cache-dir
RUN pip install "dvc[s3]"

# Copy application code
COPY ./ ./

# initialise dvc
RUN dvc init --no-scm -f
# configuring remote server in dvc
RUN dvc remote add -d storage s3://sentiment-intelsense/dvc-test

# Pull the trained model
RUN dvc pull 

# Verify files
RUN ls -la

# Test the Lambda handler
RUN python lambda_handler.py

# Set permissions
RUN chmod -R 0755 $MODEL_DIR

# Specify the Lambda handler
CMD ["lambda_handler.lambda_handler"]