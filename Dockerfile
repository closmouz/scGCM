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

##FROM rapidsai/miniforge-cuda:cuda12.2.2-base-ubuntu20.04-py3.10
#FROM nvidia/cuda:11.8.0-base-ubuntu20.04
##RUN apt-get update && \
##    apt-get install -y python3.8 python3-pip && \
##    apt-get clean && \
##    rm -rf /var/lib/apt/lists/*
#RUN conda create -n py38 python=3.8.4 -y
#
##RUN apt-get update
##RUN apt-get install -y meson ninja-build
#RUN conda run -n py38 apt-get install build-essential -y
#
##RUN apt-get install -y meson ninja-build build-essential python3-dev libffi-dev libhdf5-dev
#RUN conda run -n py38 pip install --upgrade pip setuptools wheel
#RUN echo "Copying files"
#COPY . /apps/
#WORKDIR /apps
#
#RUN conda run -n py38 conda install h5py==3.11.0
#RUN conda run -n py38 pip install --no-cache-dir -r try.txt
#
##ENTRYPOINT ["python","train.py"]
#


#
#FROM rapidsai/miniforge-cuda:cuda11.8.0-base-ubuntu20.04-py3.10
#
#
#RUN apt-get update
#
#RUN apt-get install build-essential -y
#
#RUN echo "Copying files"
#COPY . /apps/
#RUN conda install conda=23.1.0
##COPY environment.yml /tmp/environment.yml
##RUN mamba env create -f /tmp/environment.yml
##RUN echo "Environment ready"
##RUN echo "source activate scCGM" > ~/.bashrc
##ENV PATH /opt/conda/envs/scCGM/bin:$PATH
#WORKDIR /apps
#
#
#RUN conda install sfe1ed40::scikit-misc -y
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip3 install leidenalg
##
###RUN pip install https://data.pyg.org/whl/torch-2.4.0+cu121/torch_cluster-1.6.3+pt24cu121-cp310-cp310-win_amd64.whl
###RUN pip install https://data.pyg.org/whl/torch-2.4.0+cu121/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
###RUN pip install https://data.pyg.org/whl/torch-2.4.0+cu121/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
###RUN pip install https://data.pyg.org/whl/torch-2.4.0+cu121/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
###RUN conda install h5py==3.11.0
###RUN pip install --no-cache-dir -r try.txt
##
#ENTRYPOINT ["python","train.py"]
