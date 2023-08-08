#Our image will start on tensorflow with GPU support
FROM tensorflow/tensorflow:2.12.0-gpu
#We will we working on the home directory
WORKDIR /home
#Updating packages
RUN apt update
#Installing APT utils
RUN apt install -y apt-utils
# Installing NANO
RUN apt install -y nano
#Installing CMAKE
RUN apt install -y cmake
#Updating pip
RUN pip install --upgrade pip
RUN pip install Cython
#Installing TensorFlow Datasets
RUN pip install tensorflow-datasets
#Installing necessary packages
RUN pip install 'tensorflow-probability==0.17.0'
RUN pip install 'gudhi~=3.5.0'
RUN pip install 'numpy~=1.22.4'
RUN pip install 'imageio~=2.19.5'
RUN pip install 'matplotlib~=3.5.2'
RUN pip install 'seaborn~=0.12.2'
RUN pip install 'mediapy~=1.0.3'
RUN pip install 'tabulate~=0.8.10'
RUN pip install 'giotto-ph~=0.2.2'
RUN pip install scikit-posthocs
RUN pip install networkx
# Installing LaTeX to render the plots
RUN apt install -y texlive-latex-extra

