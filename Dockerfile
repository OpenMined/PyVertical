FROM python:3.7

LABEL version="0.2.0"
LABEL maintainer="OpenMined"

COPY . /pyvertical
WORKDIR /pyvertical

# Setup environment
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install jupyterlab

# Expose port for jupyter lab
EXPOSE 8888

# Enter into jupyter lab
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
