FROM rapidsai/miniforge-cuda:cuda11.8.0-base-ubuntu20.04-py3.10
RUN apt-get update
RUN apt-get install build-essential -y
RUN echo "Copying files"
COPY . /apps/
RUN conda install conda=23.1.0
WORKDIR /apps
RUN conda install sfe1ed40::scikit-misc -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install leidenalg
ENTRYPOINT ["python","train.py"]
