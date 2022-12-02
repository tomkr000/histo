# download and use an existing image which already has Conda
# installed and set up
FROM continuumio/miniconda3:4.7.12

# Dumb init minimal init system intended to be used in Linux containers
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init

# Because miniconda3 image releases may sometimes be behind we
# need to play catchup manually
RUN conda update conda && conda install "conda=4.8.3"

ENV CONDA_ENV_NAME conda-env

LABEL maintainer="Tom Roseberry <tom@codabiotherapeutics.com>"

# Lets get the environment up to date
RUN apt-get update && apt-get install -y --no-install-recommends

WORKDIR /coda_histo

COPY environment_docker.yml .
# Now we want to activate a Conda environment which has the
# necessary Python version installed and has all the libraries
# installed required to run our app
RUN conda env create -n $CONDA_ENV_NAME -f environment_docker.yml
RUN echo "source activate $CONDA_ENV_NAME" > /etc/bashrc
ENV PATH=/opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH
#ENV JAVA_HOME=${JAVA_HOME}:/usr/lib/jvm/default-java
#RUN pip install javabridge
#RUN pip install python-bioformats==1.5.2
