# This is implements a Docker container which mimics the travis testing
# environment with Ubuntu 12.04.
FROM ubuntu:12.04
MAINTAINER Jason K. Moore <moorepants@gmail.com>
RUN apt-get update
RUN apt-get -y install wget git
RUN apt-get -y install build-essential gfortran pkg-config
RUN apt-get -y install libblas3gf liblapack3gf liblapack-dev libmumps-dev
RUN apt-get -y install coinor-libipopt1 coinor-libipopt-dev
RUN mkdir /home/opty
RUN cd /home/opty && wget http://repo.continuum.io/miniconda/Miniconda-3.4.2-Linux-x86_64.sh -O miniconda.sh
RUN cd /home/opty && bash miniconda.sh -b -p /home/opty/miniconda
ENV PATH /home/opty/miniconda/bin:$PATH
RUN hash -r
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n test-environment python=2.7 pip numpy scipy cython nose coverage
RUN /home/opty/miniconda/envs/test-environment/bin/pip install https://github.com/sympy/sympy/releases/download/sympy-0.7.6.rc1/sympy-0.7.6.rc1.tar.gz
RUN /home/opty/miniconda/envs/test-environment/bin/pip install https://bitbucket.org/moorepants/cyipopt/get/tip.zip
RUN cd /home/opty && git clone https://github.com/csu-hmc/opty.git
RUN cd /home/opty/opty && /home/opty/miniconda/envs/test-environment/bin/python setup.py install
RUN cd /home/opty/opty && /home/opty/miniconda/envs/test-environment/bin/nosetests -v --with-coverage --cover-package=opty
