FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY


#this envs are experimental
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# install requirements
RUN pip install "dvc[s3]"
RUN pip install -r requirements_inference.txt

# initialise dvc
RUN dvc init --no-scm -f
# configuring remote server in dvc
RUN dvc remote add -d storage s3://sentiment-intelsense/dvc-test

RUN cat .dvc/config
# pulling the trained model
RUN dvc pull 

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]