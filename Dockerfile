#FROM ubuntu:18.04
# FROM nvidia/cuda:9.2-devel-ubuntu18.04 
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

#env vars for cuda
ENV CUDA_HOME /usr/local/cuda

#install miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda3 && \
    rm ~/miniconda.sh && \
    /opt/miniconda3/bin/conda clean -tipsy && \
    ln -s /opt/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo "conda deactivate && conda activate py37" >> ~/.bashrc

#https://blog.csdn.net/Mao_Jonah/article/details/89502380
COPY freeze.yml freeze.yml
RUN /opt/miniconda3/bin/conda env create -n py37 -f freeze.yml
#install pytorch
# RUN /opt/miniconda3/bin/conda  install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch -n py37
# RUN /opt/miniconda3/bin/conda  install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch -n py37
RUN /opt/miniconda3/bin/conda  install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch -n py37

#install cmake 3.13
RUN apt-get update && apt-get install -y software-properties-common
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && apt-get update && apt-get install -y cmake \
python3-distutils python-dev python3-dev \
libboost-all-dev 




#install spconv

WORKDIR /tmp/unique_for_spconv
RUN git clone --recursive https://github.com/DecaYale/spconv_plus.git
WORKDIR /tmp/unique_for_spconv/spconv_plus
RUN /opt/miniconda3/envs/py37/bin/python setup.py bdist_wheel 
RUN cd ./dist && /opt/miniconda3/envs/py37/bin/pip3 install *.whl




WORKDIR /tmp/
COPY config.jupyter.tar config.jupyter.tar
RUN tar -xvf config.jupyter.tar -C /root/


# RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update
# RUN apt-get install -y gcc-5 g++-5 
# RUN ls /usr/bin/ | grep gcc  
# # RUN ls /usr/bin/ | grep g++
# RUN mv /usr/bin/gcc /usr/bin/gcc.bak && ln -s /usr/bin/gcc-5 /usr/bin/gcc && gcc --version
# RUN mv /usr/bin/g++ /usr/bin/g++.bak && ln -s /usr/bin/g++-5 /usr/bin/g++ && g++ --version

#install apex
# RUN  . ~/.bashrc && conda activate py37 && git clone https://github.com/NVIDIA/apex.git \
# RUN . /opt/miniconda3/etc/profile.d/conda.sh && conda init bash 
# RUN ls  /opt/miniconda3/envs/py37/bin/ | grep pip
# RUN   git clone https://github.com/NVIDIA/apex.git \
#     && cd apex && git reset --hard f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 \
#     && /opt/miniconda3/envs/py37/bin/python setup.py install --cuda_ext --cpp_ext  
ENV TORCH_CUDA_ARCH_LIST "6.0 6.2 7.0 7.2"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN /opt/miniconda3/envs/py37/bin/pip3 uninstall -y apex || :
RUN /opt/miniconda3/envs/py37/bin/pip3 uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
# RUN /opt/miniconda3/envs/py37/bin/pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN /opt/miniconda3/envs/py37/bin/pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


#other pkgs
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  cmake build-essential vim xvfb unzip tmux psmisc  \
  libx11-dev libassimp-dev \
  mesa-common-dev freeglut3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#create some directories
RUN mkdir -p /home/yxu/Projects/ && ln -s /mnt/workspace/Works /home/yxu/Projects/Works \
    && mkdir -p /DATA/yxu/ && ln -s /mnt/workspace/datasets/ /DATA/yxu/LINEMOD_DEEPIM \
    && ln -s /mnt/workspace/datasets/LINEMOD/ /DATA/yxu/LINEMOD \
    && ln -s /mnt/workspace/datasets/BOP_LINEMOD/ /DATA/yxu/BOP_LINEMOD \
    && mkdir -p /mnt/lustre/xuyan2/ \
    && ln -s /home/yxu/datasets/ /mnt/lustre/xuyan2/datasets

EXPOSE 8887 8888 8889 10000 10001 10002 
WORKDIR /