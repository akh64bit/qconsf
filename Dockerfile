FROM ubuntu:16.04

MAINTAINER akhilbussiness@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgtk2.0-dev \
        git \
        tar \
        wget \
        vim && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh .
RUN chmod a+rwx Miniconda3-4.5.4-Linux-x86_64.sh && \
    bash Miniconda3-4.5.4-Linux-x86_64.sh -b && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN pip install --force-reinstall pip==9.0.3 && \
    conda create -n 36 python=3.6 -y && \
    /bin/bash -c "source activate 36" && \
    mkdir analysis
ADD . /analysis
WORKDIR /analysis   
RUN pip install -r requirements.txt &&\
    tar -xzvf bodycrop.tar.gz &&\
	cd bodycrop &&\
	mkdir models &&\
	cd models && \
    wget https://www.dropbox.com/s/2dw1oz9l9hi9avg/optimized_openpose.pb --no-check-certificate
WORKDIR /analysis

EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''
